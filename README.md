LiDAR Harmonization
--

![Pipeline Results](/src/graphics/figures/png/pipeline_results.png)

This is the lidar harmonization repo. See below for results and how to get setup.

## Setup

Conda is recommended. The `environment.yml` file contains the required dependencies. Create a new environemnt with `conda env create --file environment.yml`. The environment name is `lidar`. 

After cloning the directory:
```
cd harmonization
conda env create -f environment.yml
pip install -e .
```
to install the src module.

## Building the datasets

There are two primary datasets for this project:
- Dublin
- KY From Above

To get started with Dublin, navigate to `src/dataset`. To download dublin and produce uncompressed npy versions, run 
```
bash get_dublin
cd ..
python tools/laz_to_numpy.py
```

This will download the dublin scans into `src/dataset/dublin/laz/` and the uncompressed numpy versions into `dublin/npy/pt_src_id.npy`. The numpy versions only contain spatial coordinates (X, Y, Z), intensity, scan angle rank, surface normals (X, Y, Z), and the point source id. The numpy files will be named by point source. 

Now that we have more efficient files in place, it is time to create the training examples. These scripts are not especially organized. To generate a set of training examples, we will run a set of scripts in this order from the `dataset` directory:

```
python tools/create_dataset.py
```

The preceding scripts first query a series of neighborhoods in the overlap regions. The neighborhoods along with their ground truth centers are saved in a new dataset directory (default: `150_190000`) under `neighborhoods`. The second script will then produce a csv file listing these neighborhoods. Of particular note is that this script is also responsible for class balancing. Currently, the training examples are being undersampled by source. Then, they are oversampled by target intensity. These training examples will be divided into a train/validation/test split. They are in no way connected to each other, they simply exist somewhere in the overlap regions between the base flight (default source 1) and the other flights that overlap with the base flight.

To produce a more qualitative evaluation, we run the following scripts to generate neighborhoods from contiguous regions. By default, they pull from the overlap region from source 1 and source 37. 

`python tools/create_big_tile.py`
`python tools/create_big_tile_dataset.py`

This will create two new datasets. One is firmly in the overlap region (`big_tile_in_overlap`) and one is well outside of it (`big_tile_no_overlap`). PPTK will launch the viewer at the end of the first script, this can be used to explore the tile. Or, it can be simply be closed. 


## Training the model

Training is very straightforward. Simply run `python training/train.py` from the `src` directory. This will run training. 

## Troubleshooting

### Installing Custom Pointnet++ CUDA Kernels with Anaconda
If you are using Anaconda and you receive errors messages stating that .o files are not a recognized file format, it is because Anaconda comes with it's own `ld` linker. Find the folder `$HOME/.conda/envs/lidar/compiler_compat` and rename `ld` to `ld-old` temporarily. Then try to install the dependencies.
