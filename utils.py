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

DEFAULT_N_FEATURES = 256


class Image:
    def __init__(self, path: str, id : None =None , name : str | None =None, descr_path :str | None =None) -> None:
        self.path = os.path.abspath(path)
        self.name = name
        self.id = id
        self.descr = None
        self.nb_descr = None
        self.descr_path = descr_path
        self.size = DEFAULT_N_FEATURES

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
                i = 0
                while (chunk := file.read(4 * 128 * buffer_size)) != b"":
                    descr_format = "<" + "f" * 128
                    block = struct.iter_unpack(descr_format, chunk)
                    for des in block:
                        data[i] = (np.array(des))
                        i+=1
    
    
    def compute_descr(self,
        outfile="", nfeatures=DEFAULT_N_FEATURES
    ):
        outfile = os.path.abspath(outfile)

        self.size = nfeatures

        img = cv.imread(self.path)

        grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        sift = cv.SIFT_create(nfeatures=nfeatures)

        _, des = sift.detectAndCompute(grayscale, None)
        self.descr = np.array(des)
        self.descr.reshape((self.size,128))

        if outfile != "":
            # format de sortie : n : nbr de descripteur : 4 bytes, suivit de 128 * n flottants, chacun sur 4 bytes
            with open(outfile, "wb") as outfile:
                outfile.write(struct.pack("<l", len(des)))

                for d in des:
                    for f in d:
                        outfile.write(
                            struct.pack("<f", f)
                        )  # c'est très probablement des flottants 32 bits donc f est ok



class Database:
    def __init__(self, dir_path : str = "", auto_init = True, nb_descr_per_img=DEFAULT_N_FEATURES) -> None:
        self.dir_path = os.path.abspath(dir_path)
        self.images = None
        self.nb_descr_per_img = nb_descr_per_img
        self.name = dir_path.split('/')[-1]
        self.descr_path = self.dir_path + "../" + f"_descr_{self.nb_descr_per_img}" + self.name

    def of_images(self, arr):
        pass

    def load_images(self) : 
        assert self.dir_path != None
        files_paths = [
            self.dir_path + "/" + f
            for f in os.listdir(self.dir_path)
            if os.path.isfile(self.dir_path + "/" + f)
        ]
        self.images = np.empty((len(files_paths), self.nb_descr_per_img, 128), dtype=Image)
        files_names = [
            f
            for f in os.listdir(self.dir_path)
            if os.path.isfile(self.dir_path + "/" + f)
        ]
        descr_dir_exist = os.path.isdir(self.descr_path)
        for i,f_path in enumerate(files_paths):
            self.images[i] = Image(f_path)
            self.images[i].name = files_names[i]
            if descr_dir_exist :
                self.images[i].descr_path = self.descr_path + '/' + files_names[i]
    
            
            
    def load_descriptors(self, verbose=False):
        assert self.images != None
        assert self.dir_path != None
        if verbose:
            it = tqdm(range(len(self.images)))
        else:
            it = range(len(self.images))
        for i in it:
            self.images[i].load_descr()

    def compute_descr(self, save : bool = False, verbose = False):
        assert self.images != None
        if verbose:
            it = tqdm(self.images)
        else:
            it = self.images
        
        
        os.mkdir(self.descr_path)
        for im in it :
            outfile = ""
            if save : 
                outfile = self.descr_path + '/' + im.name
            im.compute_descr(outfile = outfile, nfeatures = self.nb_descr_per_img)


    def auto_init(self, verbose = False) :
        self.load_images()
        if os.path.isdir(self.descr_path) :
            self.load_descriptors(verbose=verbose)
        else :
            self.compute_descr(save=True, verbose=verbose)


""" def compute_descriptor_dir(
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

 """
""" if __name__ == "__main__":
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
 """