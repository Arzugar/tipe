#!/bin/python3

from utils import *
import numpy.linalg as la
import heapq as hp
import collections as cl
from scipy.spatial import KDTree
import random as rd


def second_closest_ratio(h, max_ratio):
    if len(h) == 0:
        return False
    d1, first_im = h[0]
    for d2, snd_im in h:
        if first_im.group_id != snd_im.group_id:
            return d1 <= max_ratio * d2
    return False


def query(
    data: Database,
    query_im: Image,
    search_func,
    im_k: int = 5,
    descr_k: int = 20,
    verbose=False,
    weight=lambda x: 1,
    snd_closest_ratio=True,
    max_ratio: float = 0.75,
    ignore_self=True,
):
    """
    @param data : La database dans laquelle chercher
    @param im : l'image query
    @param search_func : la fonction de recherche des plus proches voisins à appliquer sur chacun des descripeurs
    @param im_k : le nombre d'images voisines à renvoyer
    @param descr_k : le nombre de voisins maximum à considérer pour chaque descripteurs
    @param weight : la fonction (de la distance) de pondération à utiliser pour le vote

    """
    histogram = dict()
    if verbose:
        it = tqdm(query_im.descr, desc="Calcul des plus proches voisins")
    else:
        it = query_im.descr
    for query_descr in it:
        h = search_func(data, query_descr, descr_k)
        if ignore_self:
            h = list(filter(lambda x: x[1].id != query_im.id, h))

        # skip this descriptor if not relevant enought
        if snd_closest_ratio and not second_closest_ratio(h, max_ratio):
            continue

        for dist, im in h:
            # incrémente si exist déjà, sinon met à 1 * weight
            histogram[im] = histogram.get(im, 0) + weight(dist)

    return sorted(histogram.items(), key=lambda x: x[1], reverse=True)[:im_k]


def basic_search(data: Database, query_descr, descr_k: int):
    h = []
    for d, im in data.iter_descr():
        # distance euclidiènne entre les deux vecteurs
        dist = la.norm(query_descr - d)
        if len(h) < descr_k:
            hp.heappush(h, (-dist, im))
        else:
            hp.heappushpop(h, (-dist, im))
    return [(-x, y) for x, y in h]


def query_on_tree(data: Database, tree, descr_k, query_descr):
    d, inds = tree.query(query_descr, k=descr_k, p=2)
    inds = map(lambda x: data.image_of_descr_index(x), inds)
    return zip(d, inds)


def kd_tree_search_func_gen(data: Database, verbose=False):
    print("Building tree...")
    d = data.to_array()
    tree = KDTree(d, leafsize=10, balanced_tree=True)
    print("Tree built succesfully !")
    return lambda _, query_descr, descr_k: query_on_tree(
        data, tree, descr_k, query_descr
    )


if __name__ == "__main__":
    args = sys.argv

    assert len(args) == 2
    datapath = args[1]
    # impath = args[2]
    d = Database(datapath, auto_init=True, verbose=True)

    query_im = rd.choice(d.images)
    print("Image recherchée : ", query_im.name)

    search_f = kd_tree_search_func_gen(d, verbose=True)
    # search_f = basic_search
    result = query(
        d,
        query_im,
        search_f,
        im_k=5,
        descr_k=20,
        verbose=True,
        weight=lambda x: 1 / (x + 0.001),
    )
    for r in result:
        print(r[0].name, r[1])
