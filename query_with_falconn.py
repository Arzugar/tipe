#!/bin/python3

import falconn as fa
from utils import *

falconn_number_of_tables = 30

falconn_default_index_params = fa.LSHConstructionParameters()
falconn_default_index_params.dimension = DESCRIPTORS_SIZE + 1
falconn_default_index_params.lsh_family = fa.LSHFamily.Hyperplane
falconn_default_index_params.distance_function = fa.DistanceFunction.EuclideanSquared
falconn_default_index_params.l = falconn_number_of_tables
# for sparse data set it to 2 , for dense data : 1
falconn_default_index_params.num_rotations = 1
falconn_default_index_params.seed = 5721840
# we want to use all the available threads to set up
falconn_default_index_params.num_setup_threads = 0
falconn_default_index_params.storage_hash_table = (
    fa.StorageHashTable.BitPackedFlatHashTable
)
# we build 18-bit hashes so that each table has
# 2^18 bins; this is a good choise since 2^18 is of the same
# order of magnitude as the number of data points
fa.compute_number_of_hash_functions(18, falconn_default_index_params)


def falconn_init_index(d: Database, params=falconn_default_index_params):
    index = fa.LSHIndex(params)
    index.setup(d.array_of_descr)
    return index


def falconn_query_image(index, query_im, k, specific_params):
    if specific_params is not None:
        num_probes = specific_params["num_probes"]
        max_num_candidates = specific_params["max_num_candidates"]
    else:
        num_probes = -1
        max_num_candidates = -1
    distances, neighbors = [[]] * query_im.nb_descr, [[]] * query_im.nb_descr
    query_object = index.construct_query_object(
        num_probes=num_probes, max_num_candidates=max_num_candidates
    )
    for i, d in enumerate(query_im.descr):
        k_n = query_object.find_k_nearest_neighbors(d, k)
        dists = [la.norm(x - d, ord=2, axis=0) ** 2 for x in k_n]
        distances[i] = dists
        neighbors[i] = k_n
    return distances, neighbors


if __name__ == "__main__":
    dpath = "./image_data/very_small"
    d = Database(dpath, nb_descr_per_img=10)
    index = falconn_init_index(d)
    query_im = d.images[0]
    print(query_im.name)
    des = query_im.descr

    r = falconn_query_image(index, query_im, 10, None)
    print(r[0][0])
    print(r[0][1])
# a = query_object.get_candidates_with_duplicates(query_im.descr[0])
