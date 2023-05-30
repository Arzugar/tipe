#!/bin/python3

from utils import *
from scipy.spatial import cKDTree
import numpy as np


def scipy_init_index(d: Database, params=None):
    tree = cKDTree(d.array_of_descr)
    return tree


def scipy_query_image(index, query_im, k, specific_params=None):
    distances, neighbors = [[]] * query_im.nb_descr, [[]] * query_im.nb_descr
    for i, d in enumerate(query_im.descr):
        dists, indices = index.query(d, k)
        distances[i] = dists
        neighbors[i] = indices
    return distances, neighbors


if __name__ == "__main__":
    dpath = "./image_data/very_small"
    d = Database(dpath, nb_descr_per_img=10)
    index = scipy_init_index(d)
    query_im = d.images[0]
    print(query_im.name)
    des = query_im.descr

    r = scipy_query_image(index, query_im, 10, None)
    print(np.round(r[0][0], 2))
    print(r[1][0])