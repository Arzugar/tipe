import struct
import os
from tqdm import tqdm
from math import sqrt
import numpy as np
import heapq
import cv2 as cv
import os
import sys
import struct
from tqdm import tqdm


LoadError = Exception()


class Image:
    def __init__(self, path: str, id : None =None , name : str | None =None, descr_path :str | None =None) -> None:
        self.path = path
        self.name = name
        self.id = id
        self.descr = None
        self.nb_descr = None
        self.descr_path = descr_path

    def load_descr(self, buffer_size=256):
        if self.descr_path == None:
            print("Cannot load, no descr_path provided")
            raise LoadError
        else:
            with open(self.path, "rb") as file:
                lg = file.read(4)

                self.size = struct.unpack("<l", lg)[0]
                # nombre de descripteurs de l'image
                data = np.empty((self.size,128))
                while (chunk := file.read(4 * 128 * buffer_size)) != b"":
                    descr_format = "<" + "f" * 128
                    block = struct.iter_unpack(descr_format, chunk)
                    for des in block:
                        data.append(np.array(des))


class Database:
    def __init__(self, dir_path=None, nb_descr_per_img=256) -> None:
        self.dir_path = dir_path
        self.images = None
        self.nb_descr_per_img = nb_descr_per_img

    def of_images(self, arr):
        pass

    def load_descriptors(self, verbose=False):
        assert self.dir_path != None
        files_paths = [
            self.dir_path + "/" + f
            for f in os.listdir(self.dir_path)
            if os.path.isfile(self.dir_path + "/" + f)
        ]
        files_names = [
            f
            for f in os.listdir(self.dir_path)
            if os.path.isfile(self.dir_path + "/" + f)
        ]
        
        if verbose:
            it = tqdm(files_paths)
        else:
            it = files_paths
        self.images = np.empty((len(files_paths), self.nb_descr_per_img, 128))
        for i,f_path in enumerate(it):
            self.images[i] = Image(f_path)


def load_database_descriptors(indir, verbose=False):
    files_paths = [
        indir + "/" + f for f in os.listdir(indir) if os.path.isfile(indir + "/" + f)
    ]
    files_names = [f for f in os.listdir(indir) if os.path.isfile(indir + "/" + f)]
    if verbose:
        it = tqdm(files_paths)
    else:
        it = files_paths
    data = []
    for f in it:
        data += load_descriptors(f)
    return data


DEFAULT_N_FEATURES = 256


def compute_descriptor_img(
    infile, outfile="", nfeatures=DEFAULT_N_FEATURES, safe_path=False
):
    if not safe_path:
        infile = os.path.abspath(infile)
        outfile = os.path.abspath(outfile)

    img = cv.imread(infile)

    grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create(nfeatures=nfeatures)

    _, des = sift.detectAndCompute(grayscale, None)
    if outfile != "":
        # format de sortie : n : nbr de descripteur : 4 bytes, suivit de 128 * n flottants, chacun sur 4 bytes
        with open(outfile, "wb") as outfile:
            outfile.write(struct.pack("<l", len(des)))

            for d in des:
                for f in d:
                    outfile.write(
                        struct.pack("<f", f)
                    )  # c'est très probablement des flottants 32 bits donc f est ok
    else:
        return des


def compute_descriptor_dir(
    indir, outdir, nfeatures=DEFAULT_N_FEATURES, safe_path=False, verbose=False
):
    if not safe_path:
        indir = os.path.abspath(indir)
        outdir = os.path.abspath(outdir)
    files = [
        indir + "/" + f for f in os.listdir(indir) if os.path.isfile(indir + "/" + f)
    ]
    if verbose:
        it = tqdm(files)
    else:
        it = files
    for f in it:
        try:
            if not str.endswith(f, ".jpg"):
                continue
            filename = f.split("/")[-1].split(".")[0]
            compute_descriptor_img(
                f,
                outfile=outdir + "/" + "_descr" + filename,
                nfeatures=nfeatures,
                safe_path=True,
            )
        except Exception:
            print(f"Issue occured on file : {filename}")


if __name__ == "__main__":
    # Parsing à la main des entrées, éventuellement utiliser un module pour le faire proprement
    args = sys.argv

    if len(args) >= 2:
        entree = args[1]
        entree = os.path.abspath(entree)
        entree_type = os.path.isfile(entree)
    else:
        print("No input provided")
        exit(1)
    if len(args) == 3:
        sortie = args[2]
        sortie = os.path.abspath(sortie)
        sortie_type = os.path.isfile(sortie)
        assert entree_type == sortie_type
    else:
        if entree_type:
            filename = entree.split("/")[-1].split(".")[0]
            parent_dir = "/".join(entree.split("/")[:-1])
            sortie = parent_dir + "/_descr" + filename
        else:
            print("No output directory provided")
            exit(1)

    if entree_type:
        compute_descriptor_img(entree, sortie, safe_path=True)
    else:
        compute_descriptor_dir(entree, sortie, safe_path=True, verbose=True)
