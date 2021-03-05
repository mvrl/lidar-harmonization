import code
import torch
from pathlib import Path
import numpy as np

def get_metrics(outputs):
    h_targets = torch.cat([x['metrics']['h_target'] for x in outputs])
    h_preds = torch.cat([x['metrics']['harmonization'] for x in outputs])
    h_mae = torch.mean(torch.abs(h_targets.flatten() - h_preds.flatten())).item()
    i_targets = torch.cat([x['metrics']['i_target'] for x in outputs])
    i_preds = torch.cat([x['metrics']['interpolation'] for x in outputs])
    i_mae = torch.mean(torch.abs(i_targets.flatten() - i_preds.flatten())).item()
    return h_targets, h_preds, h_mae, i_targets, i_preds, i_mae


def forward_pass(model, phase, batch, criterions, optimizer, scheduler, device="cpu"):
    optimizer.zero_grad()
    with torch.set_grad_enabled(phase == 'train'):
        batch = batch.to(device=device)
        h_target = batch[:, 0, 3].clone()
        i_target = batch[:, 1, 3].clone()

        harmonization, interpolation, ss = model(batch)

        # harmonization targets for source-source examples are the same as the
        #   interpolation targets. h_target contains `target_camera` 
        #   harmonizations!

        h_target[ss] = i_target[ss]

        # code.interact(local=locals())

        h_loss = criterions['harmonization'](harmonization.squeeze(), h_target)
        i_loss = criterions['interpolation'](interpolation.squeeze(), i_target)
        loss = h_loss + i_loss

        if phase == 'train':
            loss.backward()
            optimizer.step()
            scheduler.step()  # this scheduler is for cyclicLR, which steps per batch

        return {
                'loss': loss.detach().cpu(),
                'metrics': {
                    'h_target': h_target.detach().cpu(),
                    'harmonization': harmonization.squeeze().detach().cpu(),
                    'i_target': i_target.detach().cpu(),
                    'interpolation': interpolation.squeeze().detach().cpu()}}


