from query_methods import *
from utils import *
from query_with_scipy import *
from query_with_falconn import *
from cv2 import BFMatcher
import timeit
import matplotlib.pyplot as plt


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
def linear_vs_kd(datapath, k: int, q: int = 30):
    nb_descr_val = [10, 256]
    nb_descr_effective_val = []
    linear_times = []
    kd_tree_time = []
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
        kd_tree_time.append(scipy_search_time(d, q, k))
        nb_descr_effective_val.append(d.taille_nuage)
        print("done for ", nb_descr)
        # print("linear times : ", linear_times, sep="\n")
        # print("kd-tree times : ", kd_tree_time, sep="\n")
        # print("nb_descr_effective_val : ", nb_descr_effective_val, sep="\n")

    return linear_times, kd_tree_time, nb_descr_effective_val


def eval_linear_vs_kd():
    assert len(sys.argv) == 4
    datapath = sys.argv[1]
    k = int(sys.argv[2])
    q = int(sys.argv[3])
    linear_times, kd_tree_time, nb_descr_effective_val = linear_vs_kd(datapath, k, q)
    with open("linear_vs_kd.csv", "a") as f:
        f.write("nb_descr_effective,linear_time,kd_tree_time\n")
        for i in range(len(nb_descr_effective_val)):
            f.write(
                f"{nb_descr_effective_val[i]},{linear_times[i]},{kd_tree_time[i]}\n"
            )


# charge les données de resultats_num.csv, ne conserve que celle pour nb_descr = 2048
# affiche les temps de recherche pour lsh en fonction de nb_tables
def draw_graph(x_val: str, y_val: str, z_val: str):
    assert x_val in [
        "k",
        "nb_tables",
        "nb_probes_ceof",
        "sample_size",
    ]
    assert z_val in [
        "kd_build_time",
        "lsh_build_time",
        "kd_avg_query_time",
        "lsh_avg_query_time",
        "score",
        "",
    ]

    xs = []
    ys = []
    zs = []
    data = []
    with open("./docs/resultats_num.csv", "r") as f:
        # ignore les 2 premières lignes
        f.readline()
        f.readline()

        for line in f:
            (
                nb_descriptors,
                k,
                nb_tables,
                nb_probes_ceof,
                sample_size,
                kd_build_time,
                lsh_build_time,
                kd_avg_query_time,
                lsh_avg_query_time,
                score,
            ) = line.split(",")
            # construit une dictionnaire avec les valeurs de chaque colonne
            # ajoute ce dictionnaire à la liste data
            data.append(
                {
                    "nb_descriptors": int(nb_descriptors),
                    "k": int(k),
                    "nb_tables": int(nb_tables),
                    "nb_probes_ceof": int(nb_probes_ceof),
                    "sample_size": int(sample_size),
                    "kd_build_time": float(kd_build_time),
                    "lsh_build_time": float(lsh_build_time),
                    "kd_avg_query_time": float(kd_avg_query_time),
                    "lsh_avg_query_time": float(lsh_avg_query_time),
                    "score": float(score),
                }
            )
    # ajoutes les valeurs de x et y à xs et ys
    for d in data:
        if d["nb_descriptors"] == 2048 and d["k"] == 50:
            xs.append(d[x_val])
            ys.append(d[y_val])
            if z_val != "":
                zs.append(d[z_val])

    # génère un graphique 3D ou 2D selon si z_val est "" ou non
    if z_val == "":
        plt.scatter(xs, ys)
        plt.xlabel(x_val)
        plt.ylabel(y_val)
    else:
        # affiche une surface 3D
        """fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_trisurf(xs, ys, zs)
        ax.set_xlabel(x_val)
        ax.set_ylabel(y_val)
        ax.set_zlabel(z_val)"""
        # affiche un nuage de points 3D

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(xs, ys, zs)
        ax.set_xlabel(x_val)
        ax.set_ylabel(y_val)
        ax.set_zlabel(z_val)


if __name__ == "__main__":
    eval_linear_vs_kd()
    # draw_graph("nb_tables", "nb_probes_ceof", "score")
    # plt.show()
