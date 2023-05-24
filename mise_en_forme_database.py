#!/bin/python3

import os
import math
import sys

# path = os.path.abspath(input("Enter le chemin : "))
path = os.path.abspath(sys.argv[1])
dirs = os.listdir(path)


nb_files = sum([len(os.listdir(f)) for f in dirs])
print("Detected :", nb_files)

k = 0
nb_bits = math.floor(math.log10(nb_files)) + 1


for i, d in enumerate(dirs):
    os.chdir(d)

    for f in os.listdir("."):
        print(f)
        os.rename(f, f"../{i:04d}{str(k).zfill(nb_bits)}.jpg")
        k += 1
    os.chdir("..")
