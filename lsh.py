import numpy as np
import math
from typing import Any, Callable, Hashable, Optional, List
from utils import *


import numpy.random as rd


rng = rd.default_rng()


def datar_hash_fun(r, d=DESCRIPTORS_SIZE + 1):
    a = rng.standard_normal(d)
    b = rng.uniform(low=0.0, high=r)
    return lambda x: np.floor((np.dot(x, a) + b) / r)


def datar_hash_familly(
    k: int,
    r=10,  # semble être une bonne approx
    d=DESCRIPTORS_SIZE + 1,
):
    return [datar_hash_fun(r, d) for _ in range(k)]


class HashTable:
    def __init__(self, hash_func: Callable[[Hashable], tuple]):
        self.hash_func = hash_func
        self.size = 0
        self.table = {}

    def add(self, key, value: Any) -> None:
        hash_value = self.hash_func(key)

        if hash_value not in self.table:
            self.table[hash_value] = []
        self.table[hash_value].append((key, value))

    def get(self, key: Hashable) -> Optional[Any]:
        hash_value = self.hash_func(key)
        if hash_value in self.table:
            for k, v in self.table[hash_value]:
                if k == key:
                    return v
        return None

    def get_bucket(self, key: Hashable) -> Optional[Any]:
        hash_value = self.hash_func(key)
        if hash_value in self.table:
            return self.table[hash_value]
        return None

    def __repr__(self) -> str:
        return (
            "\n".join([f"{type(x), type(y)}" for x, y in self.table.items()])
            + f"\nnb buckets = {len(self.table.keys())}"
        )


class Lsh:
    def __init__(
        self,
        nb_fun_per_table: int = 1,
        nb_tables: int = 1,
        hash_fun_fam: Callable = datar_hash_familly,
        r=10,
    ) -> None:
        def concat_hash(
            hash_functions: List[Callable[[Hashable], tuple]]
        ) -> Callable[[Hashable], tuple]:
            return lambda x: tuple([f(x) for f in hash_functions])

        self.tables = [
            HashTable(concat_hash(hash_fun_fam(nb_fun_per_table, r=r)))
            for _ in range(nb_tables)
        ]
        self.nb_tables = nb_tables
        self.total_sub_query_size = 0
        # print(self.tables, sep="\n")
        # Pas certain que ça ai exactement la bonne distribution

    def preprocess(self, database: Database, verbose=False):
        self.database = database
        if verbose:
            it = tqdm(database.iter_descr_and_im(), total=database.taille_nuage())
        else:
            it = database.iter_descr_and_im()
        for descr, im in it:
            for h_table in self.tables:
                h_table.add(key=descr, value=im)
        # print(self.tables, sep="\n")

    def query(self, point):
        subset = []
        for h_table in self.tables:
            bucket = h_table.get_bucket(point)
            if bucket != None:
                for x in bucket:
                    subset.append(x)

        return subset

    def query_knn(self, k: int, point):
        point_set = self.query(point)
        knn = basic_search_base(point_set=point_set, query_point=point, k=k)
        self.total_sub_query_size += len(point_set)
        return knn

    def __repr__(self) -> str:
        return "\n".join([str(h) for h in self.tables])

    def nb_buck_per_table(self):
        return sum([len(h.table.keys()) for h in self.tables]) / self.nb_tables

    """ def avg_nb_elt_per_bucket(self):
        total_indexed = self.nb_tables * self.database.taille_nuage() """
