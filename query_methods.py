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
    for d,im in data.iter_descr() : 
        dist = la.norm(query_descr - d) # distance euclidiènne entre les deux vecteurs
        if (len (h) < descr_k):
            hp.heappush(h, (-dist, im))
        else :
            hp.heappushpop(h, (-dist, im))
    return [(-x,y) for x,y in h]



def query_on_tree(data : Database, tree, descr_k, query_descr):
    d, inds = tree.query(query_descr, k=descr_k,p=2) 
    inds = map(lambda x : data.image_of_descr_index(x), inds)
    return zip(d, inds)

    


def kd_tree_search_func_gen(data : Database, verbose = False):
    print("Building tree...")
    d = data.to_array()
    tree = KDTree(d, leafsize=10, balanced_tree=True) 
    print("Tree built succesfully !")
    return lambda _,query_descr, descr_k : query_on_tree(data, tree, descr_k, query_descr)
    
    


if __name__ == "__main__" : 
    args = sys.argv

    assert len(args) == 2
    datapath = args[1]
    #impath = args[2]
    d = Database(datapath, auto_init=True, verbose= True)

    query_im = d.images[0]
    print("Image recherchée : ",query_im.name)
    
    search_f = kd_tree_search_func_gen(d, verbose=True)
    #search_f = basic_search
    result = query(d, query_im, search_f,im_k=10, descr_k=50, verbose=True, weight= lambda x:1/(x*x + 0.1))
    for r in result :
        print(r[0].name, r[1]) 
    





    