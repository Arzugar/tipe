#!/bin/python3

from utils import *
from query_methods import * 



if __name__ == "__main__" : # format de l'entrée : database, nombre de descripteurs à utiliser, nombre d'itération de test à effectuer
    args = sys.argv

    assert len(args) == 4
    datapath = args[1]
    nb_descriptors = int(args[2])
    sample_size = int(args[3])

    d = Database(datapath, auto_init=True, verbose= False, nb_descr_per_img=nb_descriptors)
    print("Taille du nuage de points : ", d.taille_nuage())

    total_good = 0

    search_f = kd_tree_search_func_gen(d, verbose=True)
    for _ in tqdm(range(sample_size)):
        query_im = rd.choice(d.images)

        result = query(d, query_im, search_f,im_k=5, descr_k=20, verbose=False, weight= lambda x:1/(x + 0.001))
        
        has_found_good_one = any([x[0].group_id == query_im.group_id for x in result])
        if has_found_good_one :
            total_good +=1
    print("Moyenne de retour : ",total_good/sample_size)
    with open("./eval_result.txt", "w") as save_file :
        save_file.write(f"exec : {args[0]}\n{total_good} / {sample_size} = {total_good/sample_size}\n")
    exit(0)
