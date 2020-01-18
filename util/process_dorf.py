import os
import time
import numpy as np
import pandas as pd
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
            brightness.append(np.fromstring(line, sep=" "))
        elif count % 6 == 0:
            intensities.append(np.fromstring(line, sep=" "))
        line = f.readline()
        count += 1


print(len(names))
print(len(scale))
print(len(brightness))
print(len(intensities))
df = pd.DataFrame()
df["names"] = names
df["scales"] = scale
df["brightness"] = brightness
df["intensities"] = intensities
df.to_json("response_functions.json")

brightness = np.stack(brightness)
intensities = np.stack(intensities)

print(brightness.shape)
print(intensities.shape)

output = np.save("response_functions.npy", np.stack([brightness, intensities]))
