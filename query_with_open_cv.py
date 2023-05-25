import cv2 as cv
from utils import *
import random as rd

FLANN_INDEX_LINEAR = 0
FLANN_INDEX_KDTREE = 1
FLANN_INDEX_KMEANS = 2
FLANN_INDEX_COMPOSITE = 3
FLANN_INDEX_LSH = 6


def init_matcher(d: Database, algo=0, index_param={}, search_param={}):
    m = cv.FlannBasedMatcher(index_param, search_param)

    for im in d.images:
        m.add(im.descr)

    return m


def query_search(d: Database, m: cv.FlannBasedMatcher = None):
    pass


if __name__ == "__main__":
    args = sys.argv

    assert len(args) == 2
    datapath = args[1]
    d = Database(datapath, auto_init=True, verbose=True)

    print("Nombre de points du nuage : ", d.taille_nuage())

    query_im = rd.choice(d.images)
    print("Classe de l'image recherchée : ", query_im.group_id)

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

    m = init_matcher(d, index_param=index_params)
    m.train()
    print("ok !")


"""     result = query(
        d,
        query_im,
        search_f,
        im_k=1,
        descr_k=50,  # bug : si le nombre de tables est >= descr_k , aucun voisins ne sont trouvés
        verbose=True,
        weight=lambda x: 1 / (x + 0.001),
        snd_closest_ratio=False,
        ignore_self=True,
    ) """
# print(result[0][0].group_id)
