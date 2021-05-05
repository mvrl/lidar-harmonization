Airborne LiDAR Intensity Harmonization
--
### [Project Page](https://davidthomasjones.me/publications/Intensity%20Harmonization) | [Video]() | [Paper]()
![Pipeline Results](/src/graphics/figures/png/pipeline_results.png)

Pytorch implementation of harmonizing a collection of airborne LiDAR scans, where each scan may have unique sensor callibrations or varying acquisition conditions. See the example above, where a collection of unharmonized scans (right) are harmonized (middle, left is ground truth).

Airborne LiDAR Intensity Harmonization<br>
[David Jones](https://davidthomasjones.me), [Nathan Jacobs](https://www.engr.uky.edu/directory/jacobs-nathan)

## What is LiDAR Harmonization?

LiDAR intensity, while an incredibly useful measurement as a learning or statistical feature, is influenced many factors. Large LiDAR campaigns, such as [KY Statewide](), require multiple flights across a large timespan. When the campaign is finished, the LiDAR is stitched together. Many times, the intensity of adjacent areas may look completely different, such as in the example below. 


<br />

![problem_def](/src/graphics/figures/png/problem_def.png)<br />


This presents problems when using intensity as a feature in tasks such as detection or classification. We present a new method based on deep learning to solve this problem by harmonizing LiDAR scans that have overlap with a specified target scan. An example of our method can be seen at the top of this README. Our harmonized version is the middle image, with the initial unharmonized collection being on the right. The ground truth is on the left.

## Setup

### Environment
Conda is recommended. The `environment.yml` file contains the required dependencies. Create a new environemnt with `conda env create --file environment.yml`. The environment name is `lidar`. 

After cloning the directory:
```
cd harmonization
conda env create -f environment.yml
pip install -e .
```
to install the src module.

### Project Configuration
We also create a project config. Simply run 
```
python config/genconfig.py
```

## Running code

To get started with Dublin, navigate to `src/datasets/dublin/data`. The following steps download the NYU DublinCity dataset, and then create uncompressed numpy files containing the original pointcloud data as well as added point sources and estimated point normals:  
```
bash get_dublin
cd ../../..   # src
python datasets/tools/laz_to_numpy.py
```

We have included premapped corruptions for each flight-swath as used in our research as `src/datasets/dublin/data/dorf.json` and `src/datasets/dublin/data/mapping.npy`. Confirm that `src/datasets/dublin/config.py` has the proper options. From the `src` directory, we can then harmonize dublin using these corruptions by simply running  
```
python run.py
```
The collection can be visualized by running 
```
python evaluation/visualize_scan
```

## Harmonizing your own LiDAR Collection
Harmonizing your own LiDAR is fairly straightforward. **Note**: you will need the original swaths in LAZ format. It is recommended to simply make a new data directory `src/datasets/<your_dataset_name>/data/laz`, where all the LiDAR scans will reside. Copy the base config file from `src/datasets/tools/config.py`, and simply fill in the blank lines. You can then follow the steps outlined in the *Running Code* section above. You will need to supply your config file in the imports section of `run.py` and `visualize_lidar_collection.py`. 

## Citatation
Please use the following citation
```
@inproceedings{jones2021harmonize,
  annotation = {remote_sensing,lidar,pointcloud,3d},
  author = {Jones, David and Jacobs, Nathan},
  booktitle = {IEEE International Geoscience and Remote Sensing Symposium (IGARSS)},
  title = {Intensity Harmonization for Airborne {LiDAR}},
  year = {2021}
}
```
