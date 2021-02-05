import os
import sys
import fileinput
import subprocess 
from itertools import groupby

from multiprocessing import Pool

try:
  NCORES = int(os.environ['SLURM_CPUS_PER_TASK'])
except:
  NCORES = 2

LAZPATH="/pscratch/nja224_uksr/ky_lidar/laz_raw"

def work(item):

  #key, row_gen = item 
  row_gen = item

  locs = [str(x) for x in row_gen]
  print(locs)

  return 0 #(key, len(locs))

# read from input
inpt = (str(x) for x in fileinput.input())

# group into tiles
def keyfunc(row):
  return row.rstrip().split(' ')[0]
tile_and_points = groupby(inpt, keyfunc)

# process each tile in parallel
p = Pool(NCORES)

#done = p.imap_unordered(work,inpt)
done = p.imap_unordered(work,tile_and_points)

for i, result in enumerate(done):

  if result:
    print(i, result, flush=True)
  else:
    pass 

