#
# match locations to tiles 
#

import os
import sys
import fileinput
import subprocess 
import fiona
from pyproj import CRS, Transformer
from multiprocessing import Pool
from shapely.geometry import shape
from rtree import index

try:
  NCORES = int(os.environ['SLURM_CPUS_PER_TASK'])
except:
  NCORES = 4

def work(row):

  items = row.rstrip().split(',')

  try:

    lat = float(items[0])
    lon = float(items[1])

    sp = proj.transform(lat, lon)

    nearest = list(tile_idx.nearest((sp[0], sp[1], sp[0], sp[1]), 1, objects=True))
    tile_id = nearest[0].object

    return tile_id, sp

  except:
    print('Input line in wrong format (possibly a header)')
    print(row)
    return None

crs_lidar = CRS.from_epsg(3091)
crs_map = CRS.from_epsg(4326)

proj = Transformer.from_crs(crs_map, crs_lidar)

tile_idx = index.Index()

with fiona.open(
  '/Kentucky_5k_PointCloudGrid.shp', 
  vfs='zip:///home/nja224/pscratch/ky_lidar/meta/KyPtCloudTileIndex.zip') as c:

  for i, feat in enumerate(c):
    bounds = shape(feat['geometry']).bounds
    tile_idx.insert(i, bounds, obj=feat['properties']['TileName'])

p = Pool(NCORES)

inpt = fileinput.input()
done = p.imap_unordered(work,inpt)

for i, row in enumerate(done):

  if row:
    tile_id, (x, y) = row
    print(tile_id, x, y, flush=True)
  else:
    pass 

