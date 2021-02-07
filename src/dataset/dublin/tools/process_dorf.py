# This builds dorf.json

import os
import time
import numpy as np
import pandas as pd
import json

start_time = time.time()

names = []
scale = []
brightness = []
intensities = []

# read DoRF
with open("dorfCurves.txt", 'r') as f:
    line = f.readline()
    count = 1
    while line:
        if count % 6 == 1:
            names.append(line)
        elif count % 6 == 2:
            scale.append(line)
        elif count % 6 == 4:
            brightness.append(line)
        elif count % 6 == 0:
            intensities.append(line)
        line = f.readline()
        count += 1


dorf_curves = {}
for idx, i in enumerate(names):
    dorf_curves[idx] = {"name" : i,
                        "scale": scale[idx],
                        "B": brightness[idx],
                        "I": intensities[idx]} 
        
with open("dorfCurves.json", 'w') as fp:
    json.dump(dorf_curves, fp, indent=4)
    