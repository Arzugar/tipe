import numpy as np
import math
from typing import Any, Callable, Hashable, Optional, List
from utils import *


import numpy.random as rd


rng = rd.default_rng()


def datar_hash_family(r, d=128):
    a = rng.standard_normal(d)
    b = rng.uniform(low=0.0, high=r)
    return lambda x: np.floor((np.dot(x, a) + b) / r)


class HashTable:
    def __init__(self, hash_func: Callable[[Hashable], tuple]):
        self.hash_func = hash_func
        self.size = 0
        self.table = {}

    def add(self, key: Hashable, value: Any) -> None:
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


class Lsh:
    def __init__(
        self, k: int, L: int, hash_fun_fam: Callable = datar_hash_family
    ) -> None:
        def concat_hash(
            hash_functions: List[Callable[[Hashable], tuple]]
        ) -> Callable[[Hashable], tuple]:
            return lambda x: tuple([f(x) for f in hash_functions])

        self.tables = [
            HashTable(concat_hash(hash_fun_fam(k))) for i in range(L)
        ]  # Pas certain que ça ai exactement la bonne distribution

    def preprocess(self, database: Database):
        for descr, im in database.iter_descr():
            for h_table in self.tables:
                h_table.add(key=descr, value=im)

    def query(self, point):
        subset = set()
        for h_table in self.tables:
            bucket = h_table.get_bucket(point)
            if bucket != None:
                for x in bucket:
                    subset.add(x)
        return subset

    def query_knn(self, k: int, point):
        return basic_search_base(self.query(point), query_descr=point, descr_k=k)
