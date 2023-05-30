#!/bin/python3

import sys
from typing import Generator, Any, List, Tuple
import os
from tqdm import tqdm
import struct
from math import sqrt
import numpy as np
import heapq
import cv2 as cv
import functools
import numpy.linalg as la
import heapq as hp


LoadError = Exception()

DEFAULT_N_FEATURES = 256

DESCRIPTORS_SIZE = 128

REVERSE_INDEX_DECAL_NEG = 10**-5
REVERSE_INDEX_DECAL_POS = 10**5


@functools.total_ordering
class Image:
    def __init__(
        self, path: str, name: str | None = None, descr_path: str | None = None
    ) -> None:
        self.path = os.path.abspath(path)
        self.name = self.path.split("/")[-1].split(".")[0]
        self.descr = np.empty(0, dtype=np.float32)
        self.descr_path = descr_path
        self.nb_descr = DEFAULT_N_FEATURES
        self.group_id = int(self.name[:4])
        self.id = 0

    def __hash__(self) -> int:
        return hash(self.path)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Image):
            return NotImplemented
        return self.path == other.path and self.nb_descr == other.nb_descr

    def __lt__(self, other):
        if not isinstance(other, Image):
            return NotImplemented
        return self.path < other.path and self.nb_descr < other.nb_descr

    def has_same_group(self, other):
        if not isinstance(other, Image):
            return NotImplemented
        return self.group_id == other.group_id

    def load_descr(self, buffer_size=256, reverse_index=False):
        if self.descr_path == None:
            print("Cannot load, no descr_path provided")
            raise LoadError
        else:
            with open(self.descr_path, "rb") as file:
                lg = file.read(4)
                self.nb_descr = struct.unpack("<l", lg)[0]
                # nombre de descripteurs de l'image

                descr_size = DESCRIPTORS_SIZE + (1 if reverse_index else 0)

                self.descr = np.empty(
                    (self.nb_descr, descr_size), dtype=np.float32
                )  # la n+1 ième dimension sert d'index inverse

                for i in range(0, self.nb_descr):
                    descr_bytes_size = 4 * (descr_size)
                    chunk = file.read(descr_bytes_size)
                    descriptor = np.array(
                        struct.unpack("<" + ("f" * (descr_size)), chunk),
                        dtype=np.float32,
                    )
                    if reverse_index:
                        descriptor[-1] = self.id * REVERSE_INDEX_DECAL_NEG
                    self.descr[i] = descriptor

    def compute_descr(
        self, save=False, outfile="", nfeatures=DEFAULT_N_FEATURES, reverse_index=False
    ):
        outfile = os.path.abspath(outfile)

        img = cv.imread(self.path)

        # reduit = cv.cvtColor(img, cv.CV_32F)
        grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        sift = cv.SIFT_create(nfeatures=nfeatures)

        _, des = sift.detectAndCompute(grayscale, None)

        nbr_effectif_features = min(len(des), nfeatures)

        if reverse_index:
            self.descr = np.append(
                des[:nbr_effectif_features],
                [[self.id * REVERSE_INDEX_DECAL_NEG]] * nbr_effectif_features,
                axis=1,
            )  # la dernière dimension sert d'index inversé
        else:
            self.descr = des[:nbr_effectif_features]

        self.nb_descr = nbr_effectif_features

        self.descr.reshape(
            (self.nb_descr, DESCRIPTORS_SIZE + (1 if reverse_index else 0)),
        )

        self.descr = np.ndarray.astype(self.descr, dtype=np.float32, copy=False)
        if save:
            # format de sortie : n : nbr de descripteur : 4 bytes, entier signé, little endiean, suivit de DESCRIPTORS_SIZE+1 * n flottants, chacun sur 4 bytes
            with open(outfile, "wb") as outfile:
                outfile.write(struct.pack("<l", self.nb_descr))

                for d in self.descr:
                    for f in d:
                        outfile.write(
                            struct.pack("<f", f)
                        )  # c'est des flottants 32 bits donc f est ok

    def normalise(self):
        self.descr /= la.norm(self.descr, axis=1).reshape(-1, 1)


class Database:
    def __init__(
        self,
        dir_path: str = "",
        auto_init=True,
        verbose=False,
        nb_descr_per_img=DEFAULT_N_FEATURES,
        normalise=False,
        reverse_index=False,
        center=False,
    ) -> None:
        self.dir_path = os.path.abspath(dir_path)
        self.images = np.empty(0)
        # nombre avec lequel ça a été calculé, ça peut être moins si y'a pas assez de features
        self.nb_descr_per_img = nb_descr_per_img
        self.name = dir_path.split("/")[-1]
        self.reverse_index = reverse_index

        _rajout = "rev_" if self.reverse_index else ""
        self.descr_path = os.path.abspath(
            f"{self.dir_path}/../_descr_{self.nb_descr_per_img}_{_rajout}{self.name}"
        )

        self.taille_nuage = None
        self.array_of_descr = np.empty(0)
        self.normalise = normalise
        self.center = center

        if auto_init:
            self.auto_init(verbose=verbose)

    def load_images(self):
        assert self.dir_path != None
        files_paths = [
            self.dir_path + "/" + f
            for f in os.listdir(self.dir_path)
            if os.path.isfile(self.dir_path + "/" + f)
        ]
        self.images = np.empty(len(files_paths), dtype=Image)

        descr_dir_exist = os.path.isdir(self.descr_path)
        for i, f_path in enumerate(files_paths):
            self.images[i] = Image(f_path)
            self.images[i].id = i
            if descr_dir_exist:
                self.images[i].descr_path = self.descr_path + "/" + self.images[i].name

    def load_descriptors(self, verbose=False):
        # assert self.images != np.empty(0)
        assert self.dir_path != None
        if verbose:
            it = tqdm(range(len(self.images)), desc="Chargement des descripteurs")
        else:
            it = range(len(self.images))
        for i in it:
            self.images[i].load_descr(reverse_index=self.reverse_index)

    def compute_descr(self, save: bool = False, verbose=False):
        # assert self.images != np.empty(0)
        if verbose:
            it = tqdm(self.images, desc="Calcul des descripteurs")
        else:
            it = self.images

        os.mkdir(self.descr_path)
        for im in it:
            outfile = self.descr_path + "/" + im.name

            im.compute_descr(
                outfile=outfile,
                save=save,
                nfeatures=self.nb_descr_per_img,
                reverse_index=self.reverse_index,
            )

    def auto_init(self, verbose=False):
        self.load_images()

        if os.path.isdir(self.descr_path):
            self.load_descriptors(verbose=verbose)
        else:
            self.compute_descr(save=True, verbose=verbose)
        self.compute_taille_nuage()

        self.compute_array_of_descr()

        if self.center:
            mean = np.mean(self.array_of_descr)
            self.mean = mean
            self.array_of_descr -= mean

        if self.normalise:
            for im in self.images:
                im.normalise

    def center_query(self, query):
        query -= self.mean

    def iter_descr_and_im(self) -> Generator[tuple[list[np.float32], Image], Any, None]:
        for im in self.images:
            for d in im.descr:
                yield (d, im)

    def compute_taille_nuage(self):
        if self.taille_nuage == None:
            self.taille_nuage = sum(x.nb_descr for x in self.images)
        return self.taille_nuage

    def compute_array_of_descr(self):
        assert self.taille_nuage != None
        self.array_of_descr = np.empty(
            (self.taille_nuage, DESCRIPTORS_SIZE + (1 if self.reverse_index else 0)),
            dtype=np.float32,
        )
        i = 0
        for im in self.images:
            for d in im.descr:
                self.array_of_descr[i] = d
                i += 1

    # détermine l'image associée au descripteur indexé descr_id

    def image_of_descr_id(self, descr_id):
        assert self.reverse_index
        return self._reverse_descr_index(self.array_of_descr[descr_id])

    def _reverse_descr_index(self, descr) -> Image:
        assert self.reverse_index
        # utilise la dernière dimension comme index inverse
        im_id = round(descr[DESCRIPTORS_SIZE] * REVERSE_INDEX_DECAL_POS)
        # ici l'arrondi est peut-être pas une idée de fou
        return self.images[im_id]

    def _normalise(self):
        for im in self.images:
            im.normalise
        self.normalise = True

    def _center(self):
        mean = np.mean(self.array_of_descr)
        self.mean = mean
        self.array_of_descr -= mean


def basic_search_base(point_set, query_point, k: int):
    h = []
    if len(point_set) == 0:
        print("Empty Pointset")
    for d, im in point_set:
        # distance euclidiènne entre les deux vecteurs
        dist = la.norm(query_point - d)
        if len(h) < k:
            hp.heappush(h, (-dist, im))
        else:
            hp.heappushpop(h, (-dist, im))
    return [(-x, y) for x, y in h]


if __name__ == "__main__":
    # test01()
    args = sys.argv
    nfeatures = DEFAULT_N_FEATURES
    if len(args) >= 2:
        entree = args[1]
        entree = os.path.abspath(entree)
        entree_type = os.path.isfile(entree)

    else:
        print("No input provided")
        exit(1)

    if len(args) == 3:
        nfeatures = int(args[2])

    if entree_type:
        im = Image(entree)
        im.compute_descr()
    else:
        d = Database(
            entree,
            auto_init=True,
            verbose=True,
            nb_descr_per_img=nfeatures,
            reverse_index=False,
            normalise=True,
            center=True,
        )
        a = d.array_of_descr
