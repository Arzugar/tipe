import struct
import os
from tqdm import tqdm
from math import sqrt
import numpy as np
import heapq


class Image:
    def __init__(self, path, descr, id) -> None:
        self.path = path
        self.descr = np.array


def load_descriptors(filepath, buffer_size=256):
    """
    Renvoie une array d'arrays, chacun de dimension 128, représentant un descripteur local
    @filepath : chemin du fichier à charger
    @buffer_size : nombre de descripteurs à charger en même temps
    """
    with open(filepath, "rb") as file:
        lg = file.read(4)

        # size = struct.unpack("<l", lg)[0] # nombre de descripteurs de l'image
        data = np.array()
        while (chunk := file.read(4 * 128 * buffer_size)) != b"":
            descr_format = "<" + "f" * 128
            block = struct.iter_unpack(descr_format, chunk)
            for des in block:
                data.append(np.array(des))
    return data


def load_database_descriptors(indir, verbose=False):
    files_paths = [
        indir + "/" + f for f in os.listdir(indir) if os.path.isfile(indir + "/" + f)
    ]
    files_names = [f for f in os.listdir(indir) if os.path.isfile(indir + "/" + f)]
    if verbose:
        it = tqdm(files_paths)
    else:
        it = files_paths
    data = []
    for f in it:
        data += load_descriptors(f)
    return data


def dist(point_a, point_b):
    return sqrt(sum([(x - y) * (x - y) for x, y in zip(point_a, point_b)]))


""" def clossest (image_descr, database) : 
    hist = [0]* len(database) / 256 # nombre de descripteurs par image
    for d in image_descr : 
        if dist(d) """
