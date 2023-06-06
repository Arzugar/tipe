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


# effectue une recherche linéaire en utilisant un tas pour garder les k plus petites distances
# moyenne le temps de recherche sur q, un entier requêtes aléatoires
def linear_search_time_basic(d: Database, q: int, k: int) -> float:
    queries = rd.choices(d.array_of_descr, k=q)
    h = []
    t1 = timeit.default_timer()

    for query in queries:
        for point in d.array_of_descr:
            # distance euclidiènne entre les deux vecteurs
            dist = la.norm(query - point)
            if len(h) < k:
                hp.heappush(h, -dist)
            else:
                hp.heappushpop(h, -dist)
    t2 = timeit.default_timer()
    return (t2 - t1) / q


def linear_search_time_basic_point_set(point_set, nb_points, q: int, k: int) -> float:
    queries = rd.choices(point_set, k=q)
    h = []
    t1 = timeit.default_timer()
    for query in queries:
        for point in point_set[:nb_points]:
            # distance euclidiènne entre les deux vecteurs
            dist = la.norm(query - point)
            if len(h) < k:
                hp.heappush(h, -dist)
            else:
                hp.heappushpop(h, -dist)
    t2 = timeit.default_timer()
    return (t2 - t1) / q


# recherche linéaire en fonction du nombre de points, utilisant les kd trees de scipy
# moyenne le temps de recherche sur q, un entier requêtes aléatoires
def kd_trees_search_time(point_set, nb_points, q: int, k: int) -> float:
    queries = rd.choices(point_set, k=q)
    kd_tree_index = cKDTree(point_set[:nb_points])
    t1 = timeit.default_timer()
    for query in queries:
        scipy_query_descr(kd_tree_index, query, k)
    t2 = timeit.default_timer()
    return (t2 - t1) / q


# charge la base de donnée et effectue les recherches linéaires et avec les kd trees de scipy
# renvoie les temps moyens de recherche pour chaque méthode, pour chaque nombre de points
def linear_vs_kd(datapath, k: int, q: int = 30):
    nb_descr_val = [256, 512, 1024, 2048, 4096]
    nb_descr_effective_val = []
    linear_times = []
    kd_tree_time = []
    d = Database(
        datapath,
        auto_init=True,
        verbose=False,
        nb_descr_per_img=nb_descr_val[-1],
        normalise=False,
        center=False,
        reverse_index=False,
    )

    for nb_descr in nb_descr_val:
        linear_times.append(linear_search_time_basic(d, q, k))
        kd_tree_time.append(kd_trees_search_time(d.array_of_descr, nb_descr, q, k))
        nb_descr_effective_val.append(d.taille_nuage)
        print("done for ", nb_descr)

    with open("linear_vs_kd.csv", "a") as f:
        f.write("nb_descr_effective,linear_time,kd_tree_time\n")
        for i in range(len(nb_descr_effective_val)):
            f.write(
                f"{nb_descr_effective_val[i]},{linear_times[i]},{kd_tree_time[i]}\n"
            )


# évalue les temps de recherche linéaire en fonction du nombre de points
# sauvegarde les résultats dans un fichier csv
def eval_linear(datapath, k=10, q=30):
    nb_descr_val = [
        256,
        512,
        1024,
        2048,
        4096,
    ]
    d = Database(
        datapath,
        auto_init=True,
        verbose=False,
        nb_descr_per_img=nb_descr_val[-1],
        normalise=False,
        center=False,
        reverse_index=False,
    )

    with open("eval_linear.csv", "a") as f:
        f.write("nb_descr_effective,linear_time\n")

    for nb_descr in nb_descr_val:
        t = linear_search_time_basic_point_set(d.array_of_descr, nb_descr, q, k)
        print("done for ", nb_descr)
        with open("eval_linear.csv", "a") as f:
            f.write(f"{nb_descr*len(d.images)},{t}\n")


def affiche_linear_vs_kd():
    # affiche les résultats dans deux graphiques cote à cote
    nb_descr_effective_val = []
    linear_times = []
    kd_tree_times = []
    with open("linear_vs_kd.csv", "r") as f:
        # ignore la premières lignes
        f.readline()

        for line in f:
            nb_descr_effective, linear_time, kd_tree_time = line.split(",")
            nb_descr_effective_val.append(int(nb_descr_effective))
            linear_times.append(float(linear_time))
            kd_tree_times.append(float(kd_tree_time))

    # affiche les résultats dans un graphique
    plt.subplot(1, 2, 1)
    plt.scatter(nb_descr_effective_val, linear_times)
    plt.xlabel("nombre de points")
    plt.ylabel("temps (s)")
    plt.title("temps de recherche linéaire")
    plt.subplot(1, 2, 2)
    plt.scatter(nb_descr_effective_val, kd_tree_times)
    plt.xlabel("nombre de points")
    plt.ylabel("temps (s)")
    plt.title("temps de recherche avec kd trees")
    plt.show()


# charge les données de resultats_num.csv, ne conserve que celle pour nb_descr = 2048
# affiche les temps de recherche pour lsh en fonction de nb_tables
# permet de choisir si une échelle logarithmique est utilisée pour chaque axe
# permet de moyenner les valeur de l'axe y pour chaque valeur de l'axe x
def draw_graph(
    x_val: str,
    y_val: str,
    z_val: str,
    log_x: bool = False,
    log_y: bool = False,
    moyenne: bool = False,
):
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

    # moyenne les valeurs de y pour chaque valeur de x
    if moyenne:
        x_val_set = set(xs)
        new_xs = list(x_val_set)
        new_ys = []
        # parcours les valeurs ys, et ajoute la moyenne des valeurs de ys pour chaque valeur de x
        for x_val in x_val_set:
            moy = 0
            nb = 0
            for i in range(len(xs)):
                if xs[i] == x_val:
                    moy += ys[i]
                    nb += 1
            new_ys.append(moy / nb)
        xs = new_xs
        ys = new_ys

    # génère un graphique 3D ou 2D selon si z_val est "" ou non
    if z_val == "":
        # affiche une courbe 2D
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
    if log_x:
        plt.xscale("log")
    if log_y:
        plt.yscale("log")

    plt.show()


# affiche le temps de construction des arbres k dimensionels en fonction du nombre de vecteurs de la base de données
def kd_build_time(datapath):
    nb_descr_val = [256, 512, 1024, 2048, 4096, 5000, 6000, 7000, 8000, 9000, 10000]
    nb_descr_effective_val = []
    kd_build_time = []
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
        t1 = timeit.default_timer()
        scipy_init_index(d)
        t2 = timeit.default_timer()
        kd_build_time.append((t2 - t1))

        nb_descr_effective_val.append(d.taille_nuage)
        print("done for ", nb_descr)
        # print("linear times : ", linear_times, sep="\n")
        # print("kd-tree times : ", kd_tree_time, sep="\n")
        # print("nb_descr_effective_val : ", nb_descr_effective_val, sep="\n")

    # génère le graphique
    plt.scatter(nb_descr_effective_val, kd_build_time)
    plt.xlabel("nb_descr_effective")
    plt.ylabel("time")
    plt.show()


# evalue les temps moyens de recherche avec des kd trees pour des vecteurs aléatoires de dimension 2 à 2048
# affiche les résultats dans un graphique
def curse_of_dim(nb_queries: int = 30, taille_nuage: int = 10**6):
    vector_sizes = [128, 256, 512, 1024]
    times = []
    # génère un nuage de point aléatoire, uniformément choisits dans [0, 1]^vector_size

    for vector_size in vector_sizes:
        # génère un nuage de vecteurs aléatoires de dimension vector_size
        # et effectue des recherches avec des kd trees
        nuage = np.random.rand(taille_nuage, vector_size)
        index = cKDTree(nuage)
        # génère des requêtes aléatoires
        queries = [np.random.rand(vector_size) for _ in range(nb_queries)]
        # effectue les recherches

        t1 = timeit.default_timer()
        for query in queries:
            scipy_query_descr(index, query, 10)
        t2 = timeit.default_timer()
        times.append((t2 - t1) / nb_queries)

    # sauvegarde les résultats dans un fichier csv
    with open("curse_of_dim.csv", "a") as f:
        f.write("vector_size,time\n")
        for i in range(len(vector_sizes)):
            f.write(f"{vector_sizes[i]},{times[i]}\n")


def affiche_curse_of_dim():
    vector_sizes = []
    times = []
    with open("curse_of_dim.csv", "r") as f:
        # ignore les 2 premières lignes
        f.readline()
        f.readline()

        for line in f:
            vector_size, time = line.split(",")
            vector_sizes.append(int(vector_size))
            times.append(float(time))

    plt.scatter(vector_sizes, times)
    plt.xlabel("vector_size")
    plt.ylabel("time")
    plt.show()


# affiche le temps de recherche linéaire en fonction du nombre de points
def affiche_lineaire():
    vals_n = []
    times = []
    with open("eval_linear.csv", "r") as f:
        # ignore la premières lignes
        f.readline()

        for line in f:
            n, time = line.split(",")
            vals_n.append(int(n))
            times.append(float(time))

    plt.scatter(vals_n, times)
    plt.xlabel("nombre de points")
    plt.ylabel("temps (s)")
    plt.title("temps de recherche linéaire en fonction du nombre de points")
    plt.show()


if __name__ == "__main__":
    # curse_of_dim(nb_queries=10)
    linear_vs_kd("./image_data/jpg2", k=10)
    # draw_graph("nb_descr", "kd_build_time", "", log_x=False, moyenne=True)
    # affiche_curse_of_dim()
    # kd_build_time("./image")
    # eval_linear("./image_data/jpg2")
    # affiche_lineaire()
