#!/bin/python3

from utils import *
from query_methods import *
import itertools
import time

time.time()


if __name__ == "__main__":
    # format de l'entrée : database, nombre de descripteurs à utiliser, nombre d'itération de test à effectuer

    args = sys.argv
    assert len(args) == 4
    datapath = args[1]
    nb_descriptors = int(args[2])
    sample_size = int(args[3])

    d = Database(
        datapath, auto_init=True, verbose=False, nb_descr_per_img=nb_descriptors
    )
    print("Taille du nuage de points : ", d.taille_nuage())
    print("Nombre de descr par image : ", nb_descriptors)
    print("Sample size : ", sample_size)
    descr_k_vals = [5, 20, 40]
    nb_tables_val = [10, 20, 30]  # range(2, 21, 2)
    nb_fun_per_table_val = [10]  # [1, 2, 3, 4]
    r_val = [4]  # [1, 5, 10, 20]

    for nb_tables, nb_fun_per_table, r in itertools.product(
        nb_tables_val, nb_fun_per_table_val, r_val
    ):
        start_time = time.time()

        # search_f = kd_tree_search_func_gen(d, verbose=True)
        lsh_tables, search_f = init_lsh(
            d,
            verbose=False,
            r=r,
            nb_fun_per_table=nb_fun_per_table,
            nb_tables=nb_tables,
        )

        build_time = time.time() - start_time

        avg_bucket_per_tables = lsh_tables.nb_buck_per_table()

        for descr_k in descr_k_vals:
            lsh_tables.total_sub_query_size = 0
            total_score = 0
            total_query_time = 0

            for _ in range(sample_size):
                query_im = rd.choice(d.images)
                # détermine l'image associée

                start_time = time.time()
                result = query(
                    d,
                    query_im,
                    search_f,
                    im_k=1,
                    descr_k=descr_k,
                    verbose=False,
                    weight=lambda x: 1 / (x + 0.001),
                    snd_closest_ratio=False,
                )
                total_query_time += time.time() - start_time
                if len(result) > 0:
                    score = 1 if result[0][0].group_id == query_im.group_id else 0
                    total_score += score
            avg_subquery_size = lsh_tables.total_sub_query_size / sample_size

            print(
                f"Param : descr_k : {descr_k} | nb_tables : {nb_tables} | nb_fun_per_table : {nb_fun_per_table} | r : {r}"
            )
            print(
                f"Result: build_time : {build_time} | avg_bucket_per_table : {avg_bucket_per_tables} | avg_subquery_size : {avg_subquery_size} | avg_score : {total_score/sample_size} | avg_query_time : {total_query_time/sample_size}\n"
            )
