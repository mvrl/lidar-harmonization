import sys
import glob
import subprocess 
from statistics import mean
import uuid

from multiprocessing import Pool

LAZPATH="/pscratch/nja224_uksr/ky_lidar/ftp.kymartian.ky.gov/kyaped/LAZ"

#cmd_tmpl = "las2las -i {} -olaz -target_epsg 4326 -stdout -subseq 0 20 | las2las -stdin -otxt -oparse z -stdout"
#cmd_tmpl = "las2las -epsg 3091 -i {} -olaz -target_epsg 4326 -stdout | las2las -stdin -otxt -oparse z -stdout"
#cmd1_tmpl = "las2las -epsg 3091 -i {} -olaz -target_epsg 4326 -subseq 0 20 -o ./tmp.laz"
#cmd1_tmpl = "las2las -epsg 3091 -i {} -olaz -target_epsg 4326 -keep_random_fraction 0.01 -o ./tmp.laz"
#cmd1_tmpl = "las2las -epsg 3091 -i {} -olaz -target_epsg 4326 -keep_every_nth 20000 -o ./tmp.laz"
cmd1_tmpl = "las2las -epsg 3091 -i {} -olaz -target_epsg 4326 -keep_random_fraction 0.01 -o ./{}.laz"
cmd2_tmpl = "las2las -i ./{}.laz -otxt -oparse z -stdout"
cmd3_tmpl = "rm ./{}.laz"

def work(laz):

  unique_filename = str(uuid.uuid4())

  cmd = cmd1_tmpl.format(laz, unique_filename)
  #print(cmd)
  p = subprocess.run(cmd, shell=True, capture_output=True)

  if p.returncode != 0:
    print(p.stderr)

  cmd = cmd2_tmpl.format(unique_filename)
  #print(cmd)
  p = subprocess.run(cmd, shell=True, capture_output=True)

  if p.returncode != 0:
    print(p.stderr)

  avg = mean(float(n) for n in p.stdout)

  cmd = cmd3_tmpl.format(unique_filename)
  p = subprocess.run(cmd, shell=True, capture_output=True)

  if p.returncode != 0:
    print(p.stderr)

  return laz, avg

p = Pool(20)

g = glob.glob(LAZPATH + '/*.laz')

for i, (laz, avg) in enumerate(p.imap_unordered(work,g)):
  print(i, laz, avg) 

