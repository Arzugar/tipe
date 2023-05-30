from utils import *
import itertools
import time
import csv
from query_with_falconn import *
from query_with_scipy import *
import random as rd
import sys


k_val = [5, 10, 20, 50]
nb_descr_val = [256, 512, 1024, 2048]
nb_tables_val = [10, 20, 30, 40, 50, 60]
nb_probes_coef_val = range(1, 21, 2)


args = sys.argv
assert len(args) == 2
datapath = args[1]

for nb_descr in nb_descr_val:
    d = Database(
        datapath,
        auto_init=True,
        verbose=False,
        nb_descr_per_img=nb_descr,
        normalise=False,
        center=False,
        reverse_index=False,
    )
    sample_size = len(d.images) // 100  # type: ignore

    queries = rd.choices(d.images, k=sample_size)

    kd_tree_index = scipy_init_index(d)

    answers = [scipy_query_image(index, q, k=max(k_val))[1] for q in queries]

    for k, nb_tables, nb_probes_coef in itertools.product(
        k_val, nb_tables_val, nb_probes_coef_val
    ):
        params = falconn_default_index_params
        params.seed = rd.randint(0, 2**32)
        params.dimension = DESCRIPTORS_SIZE + (1 if d.reverse_index else 0)
        params.l = nb_tables

        fa.compute_number_of_hash_functions(round(np.log2(d.taille_nuage)), params)  # type: ignore

        lsh_index = falconn_init_index(d, params=params)

        results = [
            falconn_query_image(
                lsh_index,
                q,
                k,
                specific_params={"num_probes": nb_probes_coef * nb_tables},
            )[1]
            for q in queries
        ]

        intersections_card = [
            len(set(x).intersection(set(y))) for x, y in zip(results, answers)
        ]

        scores = [x / len(y) for x, y in zip(intersections_card, answers)]

        total_score = np.mean(scores)
        print(total_score)
