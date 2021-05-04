LiDAR Harmonization
--
### [Project Page](https://davidthomasjones.me/publications/Intensity%20Harmonization) | [Video]() | [Paper]()
![Pipeline Results](/src/graphics/figures/png/pipeline_results.png)

Pytorch implementation of harmonizing a collection of airborne LiDAR scans, where each scan may have unique sensor callibrations or varying acquisition conditions.

Airborne LiDAR Intensity Harmonization<br>
[David Jones](https://davidthomasjones.me), [Nathan Jacobs](https://www.engr.uky.edu/directory/jacobs-nathan)

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

This will download the dublin scans into `src/datasets/dublin/data/laz/` and the uncompressed numpy versions into `src/datasets/dublin/data/npy/pt_src_id.npy`. The numpy versions only contain spatial coordinates (X, Y, Z), intensity, scan angle rank, surface normals (X, Y, Z), and the point source id. The numpy files will be named by point source. 

We have included premapped corruptions for each flight-swath as used in our research as `src/datasets/dublin/data/dorf.json` and `src/datasets/dublin/data/mapping.npy`. Confirm that `src/datasets/dublin/config.py` has the proper options. From the `src` directory, we can then harmonize dublin using these corruptions by simply running  
```
python run.py
```
