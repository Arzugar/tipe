#!/bin/python3

from utils import *
import itertools
import timeit
import csv
from query_with_falconn import *
from query_with_scipy import *
import random as rd
import sys


def len_inter(a, b):
    i = 0
    for x in a:
        if x in b:
            i += 1
    return i


k_val = [5, 10, 15, 20, 30, 50]
nb_descr_val = [256]
nb_tables_val = [10, 20, 30, 40, 50]
nb_probes_coef_val = range(11)

rd.seed(20222023)

args = sys.argv
assert len(args) == 2
datapath = args[1]

with open("resultats_num.csv", "a", newline="") as f:
    f.write(f"Database : {datapath}\n")
    writer = csv.writer(f)
    writer.writerow(
        [
            "nb_descriptors",
            "k",
            "nb_tables",
            "nb_probes_ceof",
            "sample_size",
            "kd_build_time",
            "lsh_build_time",
            "kd_avg_query_time",
            "lsh_avg_query_time",
            "score",
        ]
    )


for nb_descr in nb_descr_val:
    d = Database(
        datapath,
        auto_init=True,
        verbose=True,
        nb_descr_per_img=nb_descr,
        normalise=True,
        center=True,  # je sais pas si il le fait bien sur les images ou juste sur array_of_descr
        reverse_index=False,
    )
    sample_size = d.taille_nuage // 1000  # type: ignore

    queries = rd.choices(d.array_of_descr, k=sample_size)
    print("Building kd tree index")
    t1 = timeit.default_timer()

    kd_tree_index = scipy_init_index(d)

    t2 = timeit.default_timer()

    kd_build_time = t2 - t1
    # print("kd tree index built")
    # print("Preocessing exact queries")
    t1 = timeit.default_timer()
    answers = [scipy_query_descr(kd_tree_index, q, k=max(k_val))[1] for q in queries]
    t2 = timeit.default_timer()
    kd_avg_query_time = (t2 - t1) / sample_size
    # print("Exact queries processed")

    # print(answers[0])
    for k, nb_tables, nb_probes_coef in itertools.product(
        k_val, nb_tables_val, nb_probes_coef_val
    ):
        params = falconn_default_index_params
        params.seed = rd.randint(0, 2**32)
        params.dimension = DESCRIPTORS_SIZE + (1 if d.reverse_index else 0)
        params.l = nb_tables

        fa.compute_number_of_hash_functions(round(np.log2(d.taille_nuage)), params)  # type: ignore
        print("Building lsh index")
        t1 = timeit.default_timer()
        lsh_index = falconn_init_index(d, params=params)
        t2 = timeit.default_timer()
        lsh_build_time = t2 - t1
        print("Lsh index built")
        print("Queries on LSH ...")
        t1 = timeit.default_timer()
        results = [
            falconn_query_descr(
                lsh_index,
                q,
                k,
                specific_params={"num_probes": nb_probes_coef * nb_tables},
            )[1]
            for q in queries
        ]
        t2 = timeit.default_timer()
        avg_lsh_query_time = (t2 - t1) / sample_size
        print("Queries on LSH done !")
        scores = []
        for r, a in zip(results, answers):  # type: ignore
            scores.append(len_inter(r, a) / k)

        avg_score = np.mean(scores)
        with open("resultats_num.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    nb_descr,
                    k,
                    nb_tables,
                    nb_probes_coef,
                    sample_size,
                    kd_build_time,
                    lsh_build_time,
                    kd_avg_query_time,
                    avg_lsh_query_time,
                    avg_score,
                ]
            )
