from utils import *
import numpy as np
import numpy.linalg as la
import heapq as hp
import collections as cl




def basic_search(data : Database, query_im : Image, k : int = 5, descr_k : int = 5, verbose = False):
    histogram = dict()
    if verbose : 
        it = tqdm(query_im.descr, desc="Calcul des plus proches voisins")
    else : 
        it = query_im.descr
    for query_descr in it :
        h = []
        for im, d in data.iter_descr() : 
            dist = la.norm(query_descr - d) # distance euclidiènne entre les deux vecteurs
            if (len (h) < descr_k):
                hp.heappush(h, (-dist, im))
            else :
                hp.heappushpop(h, (-dist, im))
        for (_, im) in h :
            histogram[im] = histogram.get(im, 0) + 1 # incrémente si exist, sinon met à 1
    return sorted(histogram.items(), key = lambda x: x[1])[:k]



if __name__ == "__main__" : 
    args = sys.argv



    assert len(args) == 2
    datapath = args[1]
    #impath = args[2]
    d = Database(datapath, auto_init=True, verbose= True)

    query_im = d.images[0]

    basic_search(d, query_im, verbose=True)

        




    