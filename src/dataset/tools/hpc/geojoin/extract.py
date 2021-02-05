import os
import sys
import fileinput
import subprocess 

from multiprocessing import Pool

try:
  NCORES = int(os.environ['SLURM_CPUS_PER_TASK'])
except:
  NCORES = 4

LAZPATH="/pscratch/nja224_uksr/ky_lidar/laz_raw"

cmd_tmpl = "las2las -epsg 3091 -i {}/{}.laz -olaz -keep_circle {} {} 100 -stdout"
cmd_tmpl = "las2las -epsg 3091 -i {}/{}.laz -olaz -keep_circle {} {} 100 -o ./{}.laz"

def work(row):

  items = row.rstrip().split(' ')

  cmd = cmd_tmpl.format(LAZPATH, items[0], items[1], items[2], items[0])
  print(cmd)
  p = subprocess.run(cmd, shell=True, capture_output=True)

  if p.returncode != 0:
    print(p.stderr)

  return len(p.stdout)

p = Pool(NCORES)

inpt = fileinput.input()
done = p.imap_unordered(work,inpt)

for i, row in enumerate(done):

  if row:
    print(i, row, flush=True)
  else:
    pass 

