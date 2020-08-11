LiDAR Harmonization
--

This is the lidar harmonization repo. See below for results and how to get setup.

## Setup

Conda is recommended. The `environment.yml` file contains the required dependencies. Create a new environemnt with `conda env create --file environment.yml`. The environment name is `lidar`. 


## Building the datasets

There are two primary datasets for this project:
- Dublin
- KY From Above

To get started with Dublin, use the included download script `src/dataset/dublin/get_dublin`. This will download the dublin scans into `src/dataset/dublin/laz/` (~6 GB). 

Next, each of these can be trimmed down into a numpy file by running `python tools/laz_to_numpy.py` from within `src/dataset`. This process converts each laz file into a numpy file containing only the XYZ coordinates, the intensity, and the scan angle. Additionally, the normals are estimated and the `pt_src_id` is set. The files will be saved as `dublin/npy/pt_src_id.npy` for whichever source id is applied at the time. (96 GB)

Now that we have more efficient files in place, it is time to create the training examples. These scripts are not especially organized. To generate a set of training examples, we will run a set of scripts in this order from the `dataset` directory:

`python tools/create_dataset.py`
`python tools/make_csv.py`

The preceding scripts first query a series of neighborhoods in the overlap regions. The neighborhoods along with their ground truth centers are saved in a new dataset directory (default: `150_190000`) under `neighborhoods`. The second script will then produce a csv file listing these neighborhoods. Of particular note is that this script is also responsible for class balancing. Currently, the training examples are being undersampled by source. Then, they are oversampled by target intensity. These training examples will be divided into a train/validation/test split. They are in no way connected to each other, they simply exist somewhere in the overlap regions between the base flight (default source 1) and the other flights that overlap with the base flight.

To produce a more qualitative evaluation, we run the following scripts to generate neighborhoods from contiguous regions. By default, they pull from the overlap region from source 1 and source 37. 

`python tools/create_big_tile.py`
`python tools/create_big_tile_dataset.py`

This will create two new datasets. One is firmly in the overlap region (`big_tile_in_overlap`) and one is well outside of it (`big_tile_no_overlap`). PPTK will launch the viewer at the end of the first script, this can be used to explore the tile. Or, it can be simply be closed. 

### Important Note
To make things easier I have extended pytorch lightning's base classes to make the "qualitative" evaluation more simple. However, this requires one addition to the library's callbacks functionality. Navigate to your conda installation directory (most likely `~/.conda`)and open `pkgs/pytorch-lightning-0.8.5-py_0/site-packages/pytorch_lightning/callbacks/base.py`. Add the following to the bottom of the class:
```
# For Harmonization
def on_qual_batch_start(self, trainer, pl_module):
    pass

def on_qual_batch_end(self, trainer, pl_module):
    pass

def on_qual_start(self, trainer, pl_module):
    pass

def on_qual_end(self, trainer, pl_module):
    pass
```

Note that this will **not** break compatibility with other projects. 


## Training the model

Training is very straightforward. Simply run `python train.py` from the `src` directory. This should run training, validation, testing, and one of the qualitative datasets. One caveat at this point is that you must choose which one to run each time. Training a model produces results in the results directory (by default) under the chosen neighborhood size (e.g., `results/50/`). 

## Troubleshooting

### Installing Custom Pointnet++ CUDA Kernels with Anaconda
If you are using Anaconda and you receive errors messages stating that .o files are not a recognized file format, it is because Anaconda comes with it's own `ld` linker. Find the folder `$HOME/.conda/envs/lidar/compiler_compat` and rename `ld` to `ld-old` temporarily. Then try to install the dependencies.
