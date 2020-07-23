import pytorch_lightning as pl
from abc import ABC, abstractmethod
from typing import Callable, List, Union, Optional, Dict, Tuple, Iterable, Any
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.trainer.evaluation_loop import TrainerEvaluationLoopMixin
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import multiprocessing
import platform
from distutils.version import LooseVersion

from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.profiler import SimpleProfiler, PassThroughProfiler, BaseProfiler
from pytorch_lightning.trainer.auto_mix_precision import TrainerAMPMixin, NATIVE_AMP_AVALAIBLE
from pytorch_lightning.trainer.callback_config import TrainerCallbackConfigMixin
from pytorch_lightning.trainer.callback_hook import TrainerCallbackHookMixin
from pytorch_lightning.trainer.data_loading import TrainerDataLoadingMixin
from pytorch_lightning.trainer.deprecated_api import (
    TrainerDeprecatedAPITillVer0_9, TrainerDeprecatedAPITillVer0_10)
from pytorch_lightning.trainer.distrib_data_parallel import TrainerDDPMixin
from pytorch_lightning.trainer.distrib_parts import (
    TrainerDPMixin, _parse_gpu_ids, determine_root_gpu_device, pick_multiple_gpus, _parse_tpu_cores)
from pytorch_lightning.trainer.evaluation_loop import TrainerEvaluationLoopMixin
from pytorch_lightning.trainer.logging import TrainerLoggingMixin
from pytorch_lightning.trainer.model_hooks import TrainerModelHooksMixin
from pytorch_lightning.trainer.optimizers import TrainerOptimizersMixin
from pytorch_lightning.trainer.supporters import TensorRunningAccum
from pytorch_lightning.trainer.training_io import TrainerIOMixin
from pytorch_lightning.trainer.training_loop import TrainerTrainLoopMixin
from pytorch_lightning.trainer.training_tricks import TrainerTrainingTricksMixin
from pytorch_lightning.trainer.lr_finder import TrainerLRFinderMixin
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities import rank_zero_warn, parsing, rank_zero_info, rank_zero_only


import warnings

from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel, LightningDataParallel
from torch import distributed as dist
from src.ex_pl.extended_lightning import ExtendedLightningModule
# from src.extended_trainer.extended_callbacks import ExtendedCallback
import code

try:
    from torch.utils.data import IterableDataset
    ITERABLE_DATASET_EXISTS = True
except ImportError:
    ITERABLE_DATASET_EXISTS = False

try:
    from apex import amp
except ImportError:
    APEX_AVAILABLE = False
else:
    APEX_AVAILABLE = True

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
except ImportError:
    XLA_AVAILABLE = False
else:
    XLA_AVAILABLE = True

try:
    import horovod.torch as hvd
except (ModuleNotFoundError, ImportError):
    HOROVOD_AVAILABLE = False
else:
    HOROVOD_AVAILABLE = True

def _has_iterable_dataset(dataloader: DataLoader):
    return ITERABLE_DATASET_EXISTS and hasattr(dataloader, 'dataset') \
        and isinstance(dataloader.dataset, IterableDataset)


def _has_len(dataloader: DataLoader) -> bool:
    """ Checks if a given Dataloader has __len__ method implemented i.e. if
    it is a finite dataloader or infinite dataloader. """

    try:
        # try getting the length
        if len(dataloader) == 0:
            raise ValueError('`Dataloader` returned 0 length.'
                             ' Please make sure that your Dataloader at least returns 1 batch')
        has_len = True
    except TypeError:
        has_len = False
    except NotImplementedError:  # e.g. raised by torchtext if a batch_size_fn is used
        has_len = False

    if has_len and _has_iterable_dataset(dataloader) and LooseVersion(torch.__version__) >= LooseVersion("1.4.0"):
        rank_zero_warn(
            'Your `IterableDataset` has `__len__` defined.'
            ' In combination with multi-processing data loading (e.g. batch size > 1),'
            ' this can lead to unintended side effects since the samples will be duplicated.'
        )
    return has_len

class ExtendedTrainerCallbackHookMixin(TrainerCallbackHookMixin):
    callbacks: List[Callback] = []
    get_model: Callable = ...

    def on_qual_start(self):
        for callback in self.callbacks:
            callback.on_qual_start(self, self.get_model())


    def on_qual_end(self):
        for callback in self.callbacks:
            callback.on_qual_end(self, self.get_model())

    def on_qual_batch_start(self):
        for callback in self.callbacks:
            callback.on_qual_batch_start(self, self.get_model())


    def on_qual_batch_end(self):
        for callback in self.callbacks:
            callback.on_qual_batch_end(self, self.get_model())


class ExtendedTrainerModelHooksMixin(TrainerModelHooksMixin):
    def is_overridden(self, method_name: str, model: ExtendedLightningModule = None) -> bool:
        if model is None:
            model = self.get_model()
        super_object = ExtendedLightningModule

        if not hasattr(model, method_name):
            return False

        instance_attr = getattr(model, method_name)
        if not instance_attr:
            return False
        super_attr = getattr(super_object, method_name)

        # when code pointers are different, it was implemented
        if hasattr(instance_attr, 'patch_loader_code'):
            # cannot pickle __code__ so cannot verify if PatchDataloader
            # exists which shows dataloader methods have been overwritten.
            # so, we hack it by using the string representation
            is_overridden = instance_attr.patch_loader_code != str(super_attr.__code__)
        else:
            is_overridden = instance_attr.__code__ is not super_attr.__code__
        return is_overridden


class ExtendedTrainerDataLoadingMixin(TrainerDataLoadingMixin):
    def _reset_eval_dataloader(self, model: ExtendedLightningModule, mode: str) -> Tuple[List[Union[int, float]], List[DataLoader]]:
        """Generic method to reset a dataloader for evaluation

        Args:
            model: The current `LightningModule`
            mode: Either `'val'` or `'test'` or `'qual'`

        Returns:
            Tuple (num_batches, dataloaders)
        """
        if self.overfit_batches > 0:
            dataloaders = self.request_dataloader(getattr(model, 'train_dataloader'))
        else:
            dataloaders = self.request_dataloader(getattr(model, f'{mode}_dataloader'))

        if not isinstance(dataloaders, list):
            dataloaders = [dataloaders]

        for loader_i in range(len(dataloaders)):
            loader = dataloaders[loader_i]

            # shuffling in val and test set is bad practice
            if mode in ('val', 'test', 'qual') and hasattr(loader, 'sampler') and isinstance(loader.sampler, RandomSampler):

                # when overfitting, the dataloader should not have sampler
                if self.overfit_batches > 0:
                    rank_zero_warn('You requested to overfit but enabled training dataloader shuffling.'
                                   ' We are turning it off for you.')
                    dataloaders[loader_i] = self.replace_sampler(loader, SequentialSampler(loader.dataset))

                else:
                    rank_zero_warn(f'Your {mode}_dataloader has `shuffle=True`, it is best practice to turn'
                                   ' this off for validation and test dataloaders.')

        if any([dl is None for dl in dataloaders]):
            rank_zero_warn("One of given dataloaders is None and it will be skipped.")


        loader_num_batches = []

        # determine number of batches
        # datasets could be none, 1 or 2+
        if len(dataloaders) != 0:
            for i, dataloader in enumerate(dataloaders):
                num_batches = len(dataloader) if _has_len(dataloader) else float('inf')
                self._worker_check(dataloader, f'{mode} dataloader {i}')

                # percent or num_steps
                limit_eval_batches = getattr(self, f'limit_{mode}_batches')

                if num_batches != float('inf'):
                    self._check_batch_limits(f'limit_{mode}_batches')

                    # limit num batches either as a percent or num steps
                    if isinstance(limit_eval_batches, float):
                        num_batches = int(num_batches * limit_eval_batches)
                    else:
                        num_batches = min(len(dataloader), limit_eval_batches)

                elif limit_eval_batches not in (0.0, 1.0):
                    raise MisconfigurationException(
                        'When using an infinite DataLoader (e.g. with an IterableDataset'
                        f' or when DataLoader does not implement `__len__`) for `limit_{mode}_batches`,'
                        f' `Trainer(limit_{mode}_batches)` must be `0.0` or `1.0`.')

                if num_batches == 0 and limit_eval_batches > 0.0 and isinstance(limit_eval_batches, float):
                    min_pct = 1.0 / len(dataloader)
                    raise MisconfigurationException(
                        f'you requested to check {limit_eval_batches} of the {mode} dataloader but'
                        f' {limit_eval_batches}*{num_batches} = 0. Please increase the limit_{mode}_batches.'
                        f' Try at least limit_{mode}_batches={min_pct}'
                    )

                loader_num_batches.append(num_batches)

        return loader_num_batches, dataloaders

    def reset_qual_dataloader(self, model) -> None:
        """ Resets the qual dataloader and determines the number of batches

        Args:
            model: The current `LightningModule`
        """

        if self.is_overridden('qual_step'):
            self.num_qual_batches, self.qual_dataloaders =\
                    self._reset_eval_dataloader(model, 'qual')

    def determine_data_use_amount(self, overfit_batches: float) -> None:
        """Use less data for debugging purposes"""
        if overfit_batches > 0:
            if isinstance(overfit_batches, float) and overfit_batches > 1:
                raise ValueError('`overfit_batches` when used as a percentage must'
                                 f' be in range 0.0 < x < 1.0 but got {overfit_batches:.3f}.')
            self.limit_train_batches = overfit_batches
            self.limit_val_batches = overfit_batches
            self.limit_test_batches = overfit_batches
            self.limit_qual_batches = overfit_batches

class ExtendedTrainerEvaluationLoopMixin(TrainerEvaluationLoopMixin):

    on_qual_batch_start: Callable
    on_qual_batch_end: Callable
    on_qual_start: Callable
    on_qual_end: Callable

    @abstractmethod
    def reset_qual_dataloader(self, *args):
        """Warning: this is just an empty shell for code implemented in another class."""

    def _evaluate(
            self,
            model: ExtendedLightningModule,
            dataloaders: List[DataLoader],
            max_batches: Union[int, List[int]],
            test_mode: bool = False,
            qual_mode: bool = False):

        """ Run evaluation code.

        Args:
            model: the model to evaluate
            dataloaders: A list of PyTorch dataloaders
            max_batches: an integer or list of integers with length of the number of dataloaders. Each
            entry is the number of batches to process in the corresponding dataloader.
            test_mode:
            qual_mode:
        """

        model.zero_grad()
        model.eval()

        # copy properties for forward overrides
        self.copy_trainer_model_properties(model)

        # disable gradients to save memory
        torch.set_grad_enabled(False)

        # bookkeeping
        outputs = []

        # convert max_batches to list
        if isinstance(max_batches, int):
            max_batches = [max_batches] * len(dataloaders)

        # run validation
        for dataloader_idx, dataloader in enumerate(dataloaders):
            dl_outputs = []

            # on TPU we have to wrap it under the ParallelLoader
            if self.use_tpu:
                device = xm.xla_device(self.tpu_id)
                dataloader = xla_pl.ParallelLoader(dataloader, [device])
                dataloader = dataloader.per_device_loader(device)

            # each dataloader has a max num batches
            dl_max_batches = max_batches[dataloader_idx]

            for batch_idx, batch in enumerate(dataloader):
                if batch is None:
                    continue

                # stop short when on fast_dev_run (sets max_batch=1)
                if batch_idx >= dl_max_batches:
                    break

                # callbacks
                if test_mode:
                    self.on_test_batch_start()
                elif qual_mode:
                    self.on_qual_batch_start()
                else:
                    self.on_validation_batch_start()
                
                # -----------------
                # RUN EVALUATION STEP
                # -----------------
                if self.use_amp and NATIVE_AMP_AVALAIBLE and not self.use_tpu:
                    with torch.cuda.amp.autocast():
                        output = self.evaluation_forward(model, batch, batch_idx, dataloader_idx, test_mode)
                else:
                    output = self.evaluation_forward(model, batch, batch_idx, dataloader_idx, test_mode)

                # on dp / ddp2 might still want to do something with the batch parts
                if test_mode:
                    if self.is_overridden('test_step_end'):
                        model_ref = self.get_model()
                        with self.profiler.profile('test_step_end'):
                            output = model_ref.test_step_end(output)
                    self.on_test_batch_end()
                elif qual_mode:
                    if self.is_overridden('qual_step_end'):
                        model_ref = self.get_model()
                        with self.profiler.profile('qual_step_end'):
                            output = model_ref.qual_step_end(output)
                    self.on_qual_batch_end()
                else:
                    if self.is_overridden('validation_step_end'):
                        model_ref = self.get_model()
                        with self.profiler.profile('validation_step_end'):
                            output = model_ref.validation_step_end(output)
                    self.on_validation_batch_end()

                # track outputs for collation
                if output is not None:
                    dl_outputs.append(output)

            outputs.append(dl_outputs)
        
        eval_results = outputs

        # with a single dataloader don't pass an array
        if len(dataloaders) == 1:
            eval_results = outputs[0]

        # give model a chance to do something with the outputs (and method defined)
        if isinstance(model, (LightningDistributedDataParallel, LightningDataParallel)):
            model = model.module

        if test_mode:
            if self.is_overridden('test_end', model=model):
                # TODO: remove in v1.0.0
                eval_results = model.test_end(eval_results)
                rank_zero_warn('Method `test_end` was deprecated in v0.7 and will be removed in v1.0.'
                               ' Use `test_epoch_end` instead.', DeprecationWarning)

            elif self.is_overridden('test_epoch_end', model=model):
                eval_results = model.test_epoch_end(eval_results)

        elif qual_mode:
            if self.is_overridden('qual_end', model=model):
                # TODO: remove in v1.0.0
                eval_results = model.qual_end(eval_results)
                rank_zero_warn('Method `qual_end` was deprecated in v0.7 and will be removed in v1.0.'
                               ' Use `test_epoch_end` instead.', DeprecationWarning)

            elif self.is_overridden('qual_epoch_end', model=model):
                eval_results = model.qual_epoch_end(eval_results)

        else:
            if self.is_overridden('validation_end', model=model):
                # TODO: remove in v1.0.0
                eval_results = model.validation_end(eval_results)
                rank_zero_warn('Method `validation_end` was deprecated in v0.7 and will be removed in v1.0.'
                               ' Use `validation_epoch_end` instead.', DeprecationWarning)

            elif self.is_overridden('validation_epoch_end', model=model):
                eval_results = model.validation_epoch_end(eval_results)


        # enable train mode again
        model.train()

        # enable gradients to save memory
        torch.set_grad_enabled(True)

        return eval_results

    def run_evaluation(self, test_mode: bool = False, qual_mode: bool = False):
        # hook
        model = self.get_model()
        model.on_pre_performance_check()

        # select dataloaders
        if test_mode:
            self.reset_test_dataloader(model)

            dataloaders = self.test_dataloaders
            max_batches = self.num_test_batches

        elif qual_mode:
            self.reset_qual_dataloader(model)

            dataloaders = self.qual_dataloaders
            max_batches = self.num_qual_batches

        else:
            # val
            if self.val_dataloaders is None:
                self.reset_val_dataloader(model)

            dataloaders = self.val_dataloaders
            max_batches = self.num_val_batches


        # enable fast_dev_run without val loop
        if dataloaders is None:
            return

        # cap max batches to 1 when using fast_dev_run
        if self.fast_dev_run:
            max_batches = [1]

        # Validation/Qual/Test begin callbacks
        if test_mode:
            self.on_test_start()
        elif qual_mode:
            self.on_qual_start()
        else:
            self.on_validation_start()

        # enable disabling validation step with limit_val_batches = 0
        should_skip = sum(max_batches) == 0
        if should_skip:
            return [], []

        # run evaluation
        eval_results = self._evaluate(self.model, dataloaders, max_batches, test_mode, qual_mode)

        # enable no returns
        eval_loop_results = []

        if eval_results is not None and len(eval_results) > 0:

            # in eval, the user may return something at every validation step without final reduction
            if not isinstance(eval_results, list):
                eval_results = [eval_results]

            for result in eval_results:
                _, prog_bar_metrics, log_metrics, callback_metrics, _ = self.process_output(result)

                # add metrics to prog bar
                self.add_progress_bar_metrics(prog_bar_metrics)

                # log results of test
                if (test_mode or qual_mode) and self.is_global_zero and self.verbose_test:
                    print('-' * 80)
                    if test_mode:
                        print('TEST RESULTS')
                    if qual_mode:
                        print('QUALITATIVE TEST RESULTS')
                    pprint(callback_metrics)
                    print('-' * 80)
                
                # log metrics
                self.log_metrics(log_metrics, {})

                # track metrics for callbacks
                self.callback_metrics.update(callback_metrics)

                if len(callback_metrics) > 0:
                    eval_loop_results.append(callback_metrics)

        # hook
        model.on_post_performance_check()

        # eventual dataset reloading
        if test_mode:
            if self.reload_dataloaders_every_epoch:
                self.reset_test_dataloader(model)
        elif qual_mode:
            if self.reload_dataloaders_every_epoch:
                self.reset_qual_dataloader(model)
        else:
            # val
            if self.reload_dataloaders_every_epoch:
                self.reset_val_dataloader(model)

        # Validation/Test end callbacks
        if test_mode:
            self.on_test_end()
        elif qual_mode:
            self.on_qual_end()
        else:
            self.on_validation_end()

        return eval_loop_results, eval_results


    def evaluation_forward(self, model, batch, batch_idx, dataloader_idx, test_mode: bool = False, qual_mode: bool = False):
        # make dataloader_idx arg in validation_step optional
        args = [batch, batch_idx]

        if (test_mode and len(self.test_dataloaders) > 1) \
                or (qual_mode and len(self.qual_dataloaders) > 1) \
                or (not test_mode and len(self.val_dataloaders) > 1):
            args.append(dataloader_idx)

        # handle DP, DDP forward
        if self.use_ddp or self.use_dp or self.use_ddp2:
            output = model(*args)
            return output

        # Horovod
        if self.use_horovod and self.on_gpu:
            batch = self.transfer_batch_to_gpu(batch, hvd.local_rank())
            args[0] = batch

        # single GPU data transfer
        if self.single_gpu:
            # for single GPU put inputs on gpu manually
            root_gpu = 0
            if isinstance(self.data_parallel_device_ids, list):
                root_gpu = self.data_parallel_device_ids[0]
            batch = self.transfer_batch_to_gpu(batch, root_gpu)
            args[0] = batch

        # TPU data  transfer
        if self.use_tpu:
            batch = self.transfer_batch_to_tpu(batch, self.tpu_id)
            args[0] = batch

        # CPU, TPU or gpu step
        if test_mode:
            output = model.test_step(*args)
        elif qual_mode:
            output = model.qual_step(*args)
        else:
            output = model.validation_step(*args)

        return output

        


class ExtendedTrainer(
        Trainer, 
        ExtendedTrainerCallbackHookMixin,
        ExtendedTrainerModelHooksMixin,
        ExtendedTrainerDataLoadingMixin,
        ExtendedTrainerEvaluationLoopMixin
        ):
    def __init__(self, limit_qual_batches: Union[int, float] = 1.0, **kwargs):
        self.qual = False

        self.limit_qual_batches = _determine_limit_batches(limit_qual_batches)
        super().__init__(**kwargs)
    
    def __qual_given_model(self, model, qual_dataloaders):
        # setup hook
        if self.is_function_implemented('setup', model):
            model.setup('qual')

        # attach data
        if qual_dataloaders is not None:
            self.__attach_dataloaders(model, qual_dataloaders=qual_dataloaders)

        # run test
        # sets up testing so we short circuit to eval
        self.set_random_port(force=True)
        self.qual = True
        self.model = model
        results = self.fit(model)
        self.qual = False

        # teardown
        if self.is_function_implemented('teardown'):
            model.teardown('qual')

        return results

    def __qual_using_best_weights(self, ckpt_path, qual_dataloaders):
        model = self.get_model()
        if self.is_function_implemented('setup', model):
            model.setup('qual')

        # if user requests the best checkpoint but we don't have it, error
        if ckpt_path == 'best' and self.checkpoint_callback.save_top_k <= 0:
            raise MisconfigurationException(
                'ckpt_path is "best", but ModelCheckpoint is not configured to save the best model.')

        # load best weights
        if ckpt_path is not None:
            # ckpt_path is 'best' so load the best model
            if ckpt_path == 'best':
                ckpt_path = self.checkpoint_callback.best_model_path

            if len(ckpt_path) == 0:
                rank_zero_warn(f'.test() found no path for the best weights, {ckpt_path}. Please '
                               f'specify a path for a checkpoint .test(ckpt_path=PATH)')
                return {}

            ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(ckpt['state_dict'])

        # attach dataloaders
        if qual_dataloaders is not None:
            self.__attach_dataloaders(model, qual_dataloaders=qual_dataloaders)

        # run tests
        self.tested_ckpt_path = ckpt_path
        self.set_random_port(force=True)
        self.qual = True
        os.environ['PL_TESTING_MODE'] = '1'
        self.model = model
        results = self.fit(model)
        self.qual = False
        del os.environ['PL_TESTING_MODE']

        # teardown
        if self.is_function_implemented('teardown'):
            model_ref = self.get_model()
            model_ref.teardown('qual')

        return results

    def _run_sanity_check(self, ref_model, model):
        # run tiny validation (if validation defined)
        # to make sure program won't crash during val
        if not self.disable_validation and self.num_sanity_val_steps > 0:
            self.reset_val_dataloader(ref_model)

            # hook and callback
            ref_model.on_sanity_check_start()
            self.on_sanity_check_start()

            num_loaders = len(self.val_dataloaders)
            max_batches = [self.num_sanity_val_steps] * num_loaders
            eval_results = self._evaluate(model,
                                          self.val_dataloaders,
                                          max_batches,
                                          False)

            # allow no returns from eval
            if eval_results is not None and len(eval_results) > 0:
                # when we get a list back, used only the last item
                if isinstance(eval_results, list):
                    eval_results = eval_results[-1]
                _, _, _, callback_metrics, _ = self.process_output(eval_results)
                self.callback_metrics = callback_metrics

            self.on_sanity_check_end()

    def run_pretrain_routine(self, model: ExtendedLightningModule):
        ref_model = model
        if self.data_parallel:
            ref_model = model.module

        # give model convenience properties
        ref_model.trainer = self

        # set local properties on the model
        self.copy_trainer_model_properties(ref_model)

        # init amp. Must be done here instead of __init__ to allow ddp to work
        if NATIVE_AMP_AVALAIBLE and self.precision == 16 and not self.use_tpu:
            self.scaler = torch.cuda.amp.GradScaler()

        # log hyper-parameters
        if self.logger is not None:
            # save exp to get started
            self.logger.log_hyperparams(ref_model.hparams)
            self.logger.save()

        if self.use_ddp or self.use_ddp2:
            torch_distrib.barrier()

        # wait for all models to restore weights
        if self.on_tpu and XLA_AVAILAIBLE:
            # wait for all processes to catch up
            torch_xla.core.xla_model.rendezvous("pl.ExtendedTrainer.run_pretrain_routine")  # is this correct?

        elif self.use_horovod:
            # wait for all processes to catch up
            hvd.join()

        # register auto-rebsubmit when on SLURM
        self.register_slurm_signal_handlers()

        # print model summary
        if self.is_global_zero and self.weights_summary is not None and not (self.testing or self.qual):
            if self.weights_summary in ModelSummary.MODES:
                ref_model.summarize(mode=self.weights_summary)
            else:
                raise MisconfigurationException(
                    "weights_summary can be None, " + ", ".join(ModelSummary.MODES)
                )

        # track model now.
        # if cluster resets state, the model will update with the saved weights
        self.model = model

        # restore training and model before hpc is called
        self.restore_weights(model)

        # when testing requested only run test and return
        if self.testing:
            # only load test dataloader for testing
            # self.reset_test_dataloader(ref_model)
            eval_loop_results, _ = self.run_evaluation(test_mode=True)

            if len(eval_loop_results) == 0:
                return 1

            # remove the tensors from the eval results
            for i, result in enumerate(eval_loop_results):
                if isinstance(result, dict):
                    for k, v in result.items():
                        if isinstance(v, torch.Tensor):
                            result[k] = v.cpu().item()

            return eval_loop_results

        # when qual requested only run qual and retrun
        if self.qual:
            # only load qual dataloader for qual-testing
            eval_loop_results, _ = self.run_evaluation(test_mode= False, qual_mode=True)

            if len(eval_loop_results) == 0:
                return 1

            # remove the tensors from the eval results
            for i, result in enumerate(eval_loop_results):
                if isinstance(result, dict):
                    for k, v in result.items():
                        if isinstance(v, torch.Tensor):
                            result[k] = v.cpu().item()

            return eval_loop_results

        # check if we should run validation during training
        self.disable_validation = not (self.is_overridden('validation_step') and self.limit_val_batches > 0) \
            and not self.fast_dev_run

        # run a few val batches before training starts
        self._run_sanity_check(ref_model, model)

        # clear cache before training
        if self.on_gpu and self.root_gpu is not None:
            # use context because of:
            # https://discuss.pytorch.org/t/out-of-memory-when-i-use-torch-cuda-empty-cache/57898
            with torch.cuda.device(f'cuda:{self.root_gpu}'):
                torch.cuda.empty_cache()

        # CORE TRAINING LOOP
        self.train()


    def qualitative_test(
            self,
            model: Optional[ExtendedLightningModule]=None,
            qual_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
            chkpt_path: Optional[str] = 'best',
            verbose: bool = True):

        self.verbose_qual = verbose

        if self.global_rank != 0:
            return

        self.setup('qual')

        if model is not None:
            results = self.__qual_given_model(model, qual_dataloaders)
        else:
            results = self.__qual_using_best_weights(ckpt_path, qual_dataloaders)

        self.teardown('qual')

        return results

def _determine_limit_batches(batches: Union[int, float]) -> Union[int, float]:
    if 0 <= batches <= 1:
        return batches
    elif batches > 1 and batches % 1.0 == 0:
        return int(batches)
    else:
        raise MisconfigurationException(
            f'You have passed invalid value {batches}, it has to be in (0, 1) or nature number.'
        )

