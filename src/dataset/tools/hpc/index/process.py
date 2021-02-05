import sys
import glob
import subprocess 
from statistics import mean
import uuid

from multiprocessing import Pool

LAZPATH="/pscratch/nja224_uksr/ky_lidar/laz_raw"

cmd_tmpl = "lasindex -i {}"

def work(laz):

  cmd = cmd_tmpl.format(laz)
  p = subprocess.run(cmd, shell=True, capture_output=True)

  if p.returncode != 0:
    print(p.stderr)

  return laz

p = Pool(20)

g = glob.glob(LAZPATH + '/*.laz')

for i, laz in enumerate(p.imap_unordered(work,g)):
  print(i, laz) 

