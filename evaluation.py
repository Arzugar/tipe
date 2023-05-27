#!/bin/python3

from utils import *
from query_methods import *
import itertools
import time
import csv

# import timeit
time.time()


if __name__ == "__main__":
    # format de l'entrée : database, nombre de descripteurs à utiliser, nombre d'itération de test à effectuer

    args = sys.argv
    assert len(args) == 3
    datapath = args[1]
    # nb_descriptors = int(args[2])
    sample_size = int(args[2])

    # print("Nombre de descr par image : ", nb_descriptors)
    # print("Sample size : ", sample_size)
    descr_k_vals = range(5, 70, 5)
    nb_tables_val = range(10, 110, 10)
    num_probes_coef_val = range(1, 11, 1)
    nb_descriptors_val = [256, 512, 1024, 2048]
    # range(2, 21, 2)
    # nb_fun_per_table_val = [10, 20]  # [1, 2, 3, 4]

    with open("resultats_num.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "nb_descriptors",
                "nb_descr_k_per_query",
                "nb_tables",
                "nb_probes_ceof",
                "sample_size",
                "build_time",
                "avg_score",
                "avg_query_time",
            ]
        )
    for nb_descriptors in nb_descriptors_val:
        d = Database(
            datapath, auto_init=True, verbose=False, nb_descr_per_img=nb_descriptors
        )

        print(
            f"Pour nb_descriptors = {nb_descriptors}\nTaille du nuage de points : ",
            d.taille_nuage,
        )
        for nb_tables, num_probes_coef in itertools.product(
            nb_tables_val, num_probes_coef_val
        ):
            start_time = time.time()

            falconn_default_index_params.l = nb_tables

            index = falconn_init_index(d)

            build_time = time.time() - start_time

            for descr_k in descr_k_vals:
                total_score = 0
                total_query_time = 0

                for _ in range(sample_size):
                    query_im = rd.choice(d.images)
                    # détermine l'image associée

                    start_time = time.time()
                    result = query_generic(
                        d,
                        query_im,
                        falconn_query_image,
                        index,
                        descr_k=descr_k,
                        im_k=1,
                        ignore_self=True,
                        specific_params={
                            "max_num_candidates": -1,
                            "num_probes": num_probes_coef * nb_tables,
                        },
                    )

                    total_query_time += time.time() - start_time
                    if len(result) > 0:
                        score = 1 if result[0][0].group_id == query_im.group_id else 0
                        total_score += score

                avg_score = total_score / sample_size
                avg_query_time = total_query_time / sample_size
                with open("resultats_num.csv", "a", newline="") as f:
                    to_write = [
                        nb_descriptors,
                        descr_k,
                        nb_tables,
                        num_probes_coef,
                        sample_size,
                        build_time,
                        avg_score,
                        total_query_time / sample_size,
                    ]
                    writer = csv.writer(f)
                    writer.writerow(to_write)
    """                 print(
                        f"Param : descr_k : {descr_k} | nb_tables : {nb_tables} | nb_probes : {num_probes_coef}"
                    )
                    print(
                        f"Result: build_time : {build_time} | avg_score : {total_score/sample_size} | avg_query_time : {total_query_time/sample_size}\n"
                    ) """
