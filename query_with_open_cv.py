#!/bin/python3


import cv2 as cv
from utils import *
import random as rd

FLANN_INDEX_LINEAR = 0
FLANN_INDEX_KDTREE = 1
FLANN_INDEX_KMEANS = 2
FLANN_INDEX_COMPOSITE = 3
FLANN_INDEX_LSH = 6

ocv_default_params = None


def ocv_init_index(
    d: Database,
    index_param={
        "algorithm": FLANN_INDEX_LINEAR,
    },
    search_param={},
):
    m = cv.FlannBasedMatcher(indexParams=index_param, searchParams=search_param)
    # t = np.array([im.descr for im in d.images])
    # m.add(t)
    for im in d.images:
        m.add([im.descr])
        m.train()

    return m


def ocv_query_search(index: cv.FlannBasedMatcher, query_im, k, specific_params=None):
    r = index.knnMatch(query_im.descr, k=k)
    # print(*r[:2], sep="\n")
    return [[x.distance for x in m] for m in r], [[x.trainIdx for x in m] for m in r]


if __name__ == "__main__":
    dpath = "./image_data/very_small"
    d = Database(dpath, nb_descr_per_img=10)
    index = ocv_init_index(d)
    query_im = d.images[2]
    print(query_im.name)
    des = query_im.descr

    r = ocv_query_search(index, query_im, 10, None)
    print(r[0][0])
    print(r[1][0])
