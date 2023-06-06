from query_methods import *
from utils import *
from query_with_scipy import *
from query_with_falconn import *
from cv2 import BFMatcher
import timeit


# recherche linéaire en fonction du nombre de points, utilisant le bruteforce matcher de opencv
# moyenne le temps de recherche sur q, un entier requêtes aléatoires
def linear_search_time(d: Database, q: int, k: int) -> float:
    bf = BFMatcher()
    queries = rd.choices(d.array_of_descr, k=q)
    t1 = timeit.default_timer()
    for query in queries:
        bf.knnMatch(query, k=k)
    t2 = timeit.default_timer()
    return (t2 - t1) / q


# recherche linéaire en fonction du nombre de points, utilisant les kd trees de scipy
# moyenne le temps de recherche sur q, un entier requêtes aléatoires
def scipy_search_time(d: Database, q: int, k: int) -> float:
    queries = rd.choices(d.array_of_descr, k=q)
    kd_tree_index = scipy_init_index(d)
    t1 = timeit.default_timer()
    for query in queries:
        scipy_query_descr(kd_tree_index, query, k)
    t2 = timeit.default_timer()
    return (t2 - t1) / q


# charge la base de donnée et effectue les recherches linéaires et avec les kd trees de scipy
# renvoie les temps moyens de recherche pour chaque méthode, pour chaque nombre de points
def linear_vs_scipy(datapath, k: int, q: int = 30):
    nb_descr_val = [256, 5000, 6000, 7000, 8000, 9000, 10000, 20000, 30000]
    nb_descr_effective_val = []
    linear_times = []
    scipy_times = []
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
        linear_times.append(linear_search_time(d, q, k))
        scipy_times.append(scipy_search_time(d, q, k))
        nb_descr_effective_val.append(d.taille_nuage)
        print("done for ", nb_descr)
        print("linear times : ", linear_times, sep="\n")
        print("scipy times : ", scipy_times, sep="\n")
        print("nb_descr_effective_val : ", nb_descr_effective_val, sep="\n")

    return linear_times, scipy_times, nb_descr_effective_val


if __name__ == "__main__":
    assert len(sys.argv) == 4
    datapath = sys.argv[1]
    k = int(sys.argv[2])
    q = int(sys.argv[3])
    linear_times, scipy_times, nb_descr_effective_val = linear_vs_scipy(datapath, k, q)
    print(linear_times)
    print(scipy_times)
    print(nb_descr_effective_val)
