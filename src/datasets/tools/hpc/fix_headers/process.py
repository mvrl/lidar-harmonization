import os
import sys
import fileinput
import subprocess 

from multiprocessing import Pool

NCORES = int(os.environ['SLURM_CPUS_PER_TASK'])

RAWPATH="/pscratch/nja224_uksr/ky_lidar/laz_raw"
FIXPATH="/pscratch/nja224_uksr/ky_lidar/laz_fix"

os.makedirs(FIXPATH, exist_ok=True)

cmd_tmpl = "las2las -epsg 3089 -elevation_feet -i {} -olaz -target_epsg 3089 -target_elevation_feet -keep_z 0 8000 -o {}"

def work(laz):

  laz = laz.rstrip()

  laz_in = RAWPATH + '/' + laz
  laz_out = FIXPATH + '/' + laz

  cmd = cmd_tmpl.format(laz_in, laz_out)
  p = subprocess.run(cmd, shell=True, capture_output=True)

  if p.returncode != 0:
    print(p.stderr, flush=True)

  return laz_in, laz_out

p = Pool(8)

inpt = fileinput.input()
done = p.imap_unordered(work,inpt)

for i, (laz_in, laz_out) in enumerate(done):
  print(i, laz_in, laz_out, flush=True) 

