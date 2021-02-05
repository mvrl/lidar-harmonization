#!/bin/bash

for INFILE in $(ls ${LAZPATH}/*.laz)
do
  time las2las -i ${INFILE} -olaz -target_epsg 4326 -stdout | las2las -stdin -otxt -oparse z -stdout | awk '{s+=$1}END{print "ave:",s/NR}'
done

