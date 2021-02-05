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
import pdal

USE_LCC = True 

crs_map = CRS.from_epsg(4326)

if USE_LCC:
  crs_lidar = CRS.from_epsg(3089)
  EPT_JSON = '/pscratch/nja224_uksr/ky_lidar/entwine_all/ept.json'
else:
  crs_lidar = CRS.from_epsg(3857)
  EPT_JSON = 'https://s3-us-west-2.amazonaws.com/usgs-lidar-public/KY_FullState/ept.json'

json_tmpl = """
{{
  "pipeline":
  [
    {{
      "type": "readers.ept",
      "filename": "{EPT_JSON}",
      "bounds": "([{x_min}, {x_max}], [{y_min}, {y_max}])"
    }}
  ]
}}
"""
    #,
    #{{
    #    "type": "filters.sort",
    #    "dimension": "X"
    #}}

try:
  NCORES = int(os.environ['SLURM_CPUS_PER_TASK'])
except:
  NCORES = 4

def work(row):

  items = row.rstrip().split(',')

  try:
    lat = float(items[0])
    lon = float(items[1])
  except:
    # print('Input line in wrong format (possibly a header)')
    return None

  x,y = proj.transform(lat, lon)

  x_min, x_max = (x+offset for offset in (-50,50))
  y_min, y_max = (y+offset for offset in (-50,50))

  json_def = json_tmpl.format(x_min=x_min,x_max=x_max,y_min=y_min,y_max=y_max,EPT_JSON=EPT_JSON)

  #print(json_def)
  pipeline = pdal.Pipeline(json_def)
  pipeline.validate() # check if our JSON and options were good
  pipeline.loglevel = 0 #really noisy
  count = pipeline.execute()

  arrays = pipeline.arrays

  return count 

proj = Transformer.from_crs(crs_map, crs_lidar)

p = Pool(NCORES)

inpt = fileinput.input()
done = p.imap_unordered(work,inpt)

for i, output in enumerate(done):

  #output = work(row)

  if output:
    print(output, flush=True)
  else:
    pass 

