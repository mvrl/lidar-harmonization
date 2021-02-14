import os
import sys
import fileinput
import subprocess 

from multiprocessing import Pool

NCORES = int(os.environ['SLURM_CPUS_PER_TASK'])

cmd_tmpl = "lasindex -i {}"

def work(laz):

  laz = laz.rstrip()

  cmd = cmd_tmpl.format(laz)
  p = subprocess.run(cmd, shell=True, capture_output=True)

  if p.returncode != 0:
    print(p.stderr, flush=True)

  return laz

p = Pool(NCORES)

inpt = fileinput.input()
done = p.imap_unordered(work,inpt)

for i, laz in enumerate(done):
  print(i, laz, flush=True)

