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
                    "nb_probes_coef": int(nb_probes_ceof),
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
            if y_val != "":
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
        plt.xlabel("nombre de tables")
        plt.ylabel("temps (s)")
        plt.title(
            "temps de construction de la structure lsh en fonction du nombre de tables"
        )

    else:
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
    nb_descr_val = [256, 512, 1024, 1500, 2048, 3000, 4096]
    nb_descr_effective_val = []
    kd_build_time = []
    with open("kd_build_time.csv", "a") as f:
        f.write("nb_descr_effective,kd_build_time\n")

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
        # sauvegarde les résultats dans un fichier csv
        with open("kd_build_time.csv", "a") as f:
            for i in range(len(nb_descr_effective_val)):
                f.write(f"{nb_descr_effective_val[i]},{kd_build_time[i]}\n")


# evalue les temps moyens de recherche avec des kd trees pour des vecteurs aléatoires de dimension 2 à 2048
# affiche les résultats dans un graphique
def curse_of_dim(nb_queries: int = 30, taille_nuage: int = 10**6):
    vector_sizes = [600, 700, 800, 900]
    times = []
    # génère un nuage de point aléatoire, uniformément choisits dans [0, 1]^vector_size
    with open("curse_of_dim.csv", "a") as f:
        f.write("vector_size,time\n")
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
        delta_t = (t2 - t1) / nb_queries

        # sauvegarde les résultats dans un fichier csv
        with open("curse_of_dim.csv", "a") as f:
            f.write(f"{vector_size},{delta_t}\n")


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
    plt.xlabel("taille des vecteurs")
    plt.ylabel("temps (s)")
    plt.title("Malédiction de la dimension")
    plt.xscale("log")
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


def second_closest_ratio_test(
    d: Database, dists: List[np.float32], closests, max_ratio
):
    if len(closests) == 0:
        return True  # à voir si c'est le plus intelligent
    d1, first_im = dists[0], d.simple_reverse_index(closests[0])
    for d2, snd_descr_id in zip(dists[1:], closests[1:]):  # type: ignore
        snd_im = d.simple_reverse_index(snd_descr_id)
        if first_im.group_id != snd_im.group_id:
            return d1 <= max_ratio * d2
    return False


# à partir d'une base de donnée, met en place une structure LSH
# une image est choisie aléatoirement dans la base de données
# pour chacun des descripteurs de cette image on en trouve les k plus proches voisins
# puis chaque descripteur ainsi trouvé vote pour la classe de l'image à laquelle il appartient
# l'image qui a le plus de votes est choisie comme résultat
def lsh_vote(datapath, k: int, nb_tables: int, nb_probes_coef: int, q: int = 30):
    # on charge la base de données
    d = Database(
        datapath,
        auto_init=True,
        verbose=False,
        nb_descr_per_img=512,
        normalise=True,
        center=True,
        reverse_index=False,
    )
    queries = rd.choices(d.images, k=q)
    # on construit la structure LSH
    params = falconn_default_index_params
    params.seed = rd.randint(0, 2**32)
    params.l = nb_tables
    fa.compute_number_of_hash_functions(round(np.log2(d.taille_nuage)), params)  # type: ignore
    lsh_index = falconn_init_index(d, params=params)
    # on effectue les requêtes
    score = 0
    for query in queries:
        distances, neighbors = falconn_query_image(
            lsh_index, query, k, {"num_probes": nb_probes_coef * nb_tables}
        )
        # on compte les votes
        votes = {}
        for dists, k_closests_descr in zip(distances, neighbors):  # type: ignore
            # skip this descriptor if not relevant enought
            if not second_closest_ratio_test(d, dists, k_closests_descr, 0.75):
                continue
            for dist, descr_id in zip(dists, k_closests_descr):
                associated_im = d.simple_reverse_index(descr_id)
                if associated_im.id == query.id:
                    continue
                votes[associated_im] = votes.get(associated_im, 0) + 1 / (dist + 0.0001)
        # determine si la requête a été correctement classifiée
        if len(votes) == 0:
            continue
        sorted_votes = list(
            map(
                lambda x: x[0].group_id,
                sorted(votes.items(), key=lambda x: x[1], reverse=True),
            )
        )
        # print("votes : ", sorted_votes[:10])
        if query.group_id == sorted_votes[0]:
            score += 1

    # enregistre le score dans un fichier csv
    with open("lsh_vote.csv", "a") as f:
        f.write(f"{d.taille_nuage},{k},{nb_tables},{nb_probes_coef},{q},{score/q}\n")

    # return score / q


if __name__ == "__main__":
    # curse_of_dim(nb_queries=10)
    # linear_vs_kd("./image_data/jpg2", k=10)
    # draw_graph(    "nb_tables", "nb_probes_coef", "lsh_avg_query_time", log_x=False, moyenne=False)
    # affiche_curse_of_dim()
    # kd_build_time("./image_data/jpg2")
    # eval_linear("./image_data/jpg2")
    # affiche_lineaire()
    lsh_vote("./image_data/Wonders_of_World", 10, 30, 20, q=100)
