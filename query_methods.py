#!/bin/python3

from utils import *
import numpy.linalg as la

import numpy.random as rd
from typing import List, Tuple
from query_with_falconn import *


def second_closest_ratio_test(
    d: Database, dists: List[np.float32], closests, max_ratio
):
    if len(closests) == 0:
        return True  # à voir si c'est le plus intelligent
    d1, first_im = dists[0], closests[0]
    for d2, snd_descr_id in zip(dists[1:], closests[1:]):  # type: ignore
        snd_im = d.image_of_descr_id(snd_descr_id)
        if first_im.group_id != snd_im.group_id:
            return d1 <= max_ratio * d2
    return False


# Modifier pour appeler une fonction qui renvoie tous les descripteurs proches de ceux de l'image
# Fonction générique pour
def query(
    data: Database,
    query_im: Image,
    search_func,
    index,
    specific_params=None,
    im_k: int = 5,
    descr_k: int = 20,
    verbose=False,
    weight=lambda x: 1 / (x + 0.00001),
    snd_closest_ratio=True,
    max_ratio: float = 0.75,
    ignore_self=True,
    vote_for_class=False,
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
        print("Searching for voisins")
    distances, k_closests = search_func(index, query_im, descr_k, specific_params)
    if verbose:
        print("Voisins trouvés\nVote en cours")
    for dists, k_closests_descr in zip(distances, k_closests):  # type: ignore
        # skip this descriptor if not relevant enought
        if snd_closest_ratio and not second_closest_ratio_test(
            data, dists, k_closests_descr, max_ratio
        ):
            continue

        for dist, descr_id in zip(dists, k_closests_descr):  # type: ignore
            associated_im = d.image_of_descr_id(descr_id)
            if ignore_self and associated_im.id == query_im.group_id:
                continue

            # incrémente si existe déjà, sinon met à 1 * weight
            if vote_for_class:
                histogram[im.group_id] = histogram.get(
                    associated_im.group_id, 0
                ) + weight(dist)
            else:
                histogram[im] = histogram.get(associated_im, 0) + weight(dist)
    if verbose:
        print("Vote terminé")
    return sorted(histogram.items(), key=lambda x: x[1], reverse=True)[:im_k]


if __name__ == "__main__":
    rd.seed(1)
    args = sys.argv

    assert len(args) == 2
    datapath = args[1]
    # impath = args[2]
    d = Database(datapath, auto_init=True, verbose=True)

    print("Nombre de points du nuage : ", d.taille_nuage)

    query_im = rd.choice(d.images)
    print("Classe de l'image recherchée : ", query_im.group_id)

    index = falconn_init_index(d)

    r = query(
        d, query_im, falconn_query_image, index, snd_closest_ratio=True, verbose=True
    )
    print(r)
