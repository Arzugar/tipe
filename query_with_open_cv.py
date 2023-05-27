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


""" 
if __name__ == "__main__":
    args = sys.argv

    assert len(args) == 2
    datapath = args[1]
    d = Database(datapath, auto_init=True, verbose=True)

    print("Nombre de points du nuage : ", d.taille_nuage())

    query_im = rd.choice(d.images)
    print("Classe de l'image recherchée : ", query_im.group_id)

    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    index_params = dict(
        algorithm=FLANN_INDEX_LSH,
        trees=5,
        table_number=6,  # 12
        key_size=12,  # 20
        multi_probe_level=1,
    )
    m = init_matcher(d, index_param=index_params)
    m.train()
 """
