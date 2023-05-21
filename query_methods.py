from utils import *
import numpy as np
import numpy.linalg as la
import heapq as hp
import collections as cl
from scipy.spatial import KDTree

def query(data : Database, im : Image, search_func, im_k : int = 5, descr_k : int = 20, verbose = False, weight = lambda x : 1):
    """
    @param data : La database dans laquelle chercher
    @param im : l'image query
    @param search_func : la fonction de recherche des plus proches voisins à appliquer sur chacun des descripeurs
    @param im_k : le nombre d'images voisines à renvoyer
    @param descr_k : le nombre de voisins maximum à considérer pour chaque descripteurs
    @param weight : la fonction (de la distance) de pondération à utiliser pour le vote

    """
    histogram = dict()
    if verbose : 
        it = tqdm(query_im.descr, desc="Calcul des plus proches voisins")
    else : 
        it = query_im.descr
    for query_descr in it : 
        h = search_func(data,query_descr, descr_k) 

        for (dist, im) in h :
            histogram[im] = histogram.get(im, 0) + weight(dist) # incrémente si exist déjà, sinon met à 1 * weight

    return sorted(histogram.items(), key = lambda x: x[1], reverse=True)[:im_k]

def basic_search(data : Database, query_descr, descr_k: int):

    h = []
    for im, d in data.iter_descr() : 
        dist = la.norm(query_descr - d) # distance euclidiènne entre les deux vecteurs
        if (len (h) < descr_k):
            hp.heappush(h, (-dist, im.name, im))
        else :
            hp.heappushpop(h, (-dist, im.name, im))
    return [(-x,y) for x,y in h]




def build_kd_tree(database : Database): 
    return KDTree(data=database.images, leafsize=10, balanced_tree=True) 



def kd_tree_search_func_gen(data : Database, verbose = False):
    print("Building tree...")
    tree = build_kd_tree(data)
    print("Tree built succesfully !")
    return lambda _,query_descr, descr_k : tree.query(query_descr, k=descr_k,p=2)
    


if __name__ == "__main__" : 
    args = sys.argv

    assert len(args) == 2
    datapath = args[1]
    #impath = args[2]
    d = Database(datapath, auto_init=True, verbose= True)

    query_im = d.images[0]
    print(query_im.name)
    for im,k in basic_search(d, query_im, verbose=True):
        print(im.name, k)

        




    