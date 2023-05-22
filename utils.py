#!/bin/python3

import sys
try:
    import struct
except ImportError:
    print("Erreur : Impossible d'importer le module 'struct'.")
    sys.exit(1)

try:
    import os
except ImportError:
    print("Erreur : Impossible d'importer le module 'os'.")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("Erreur : Impossible d'importer le module 'tqdm'.")
    sys.exit(1)

try:
    from math import sqrt
except ImportError:
    print("Erreur : Impossible d'importer la fonction 'sqrt' depuis le module 'math'.")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("Erreur : Impossible d'importer le module 'numpy'.")
    sys.exit(1)

try:
    import heapq
except ImportError:
    print("Erreur : Impossible d'importer le module 'heapq'.")
    sys.exit(1)

try:
    import cv2 as cv
except ImportError:
    print("Erreur : Impossible d'importer le module 'cv2' depuis le module 'cv'.")
    sys.exit(1)

try:
    import functools
except ImportError:
    print("Erreur : Impossible d'importer le module 'functools'.")
    sys.exit(1)


LoadError = Exception()

DEFAULT_N_FEATURES = 2048

@functools.total_ordering
class Image:
    def __init__(self, path: str, name : str | None =None, descr_path :str | None =None) -> None:
        self.path = os.path.abspath(path)
        self.name = self.path.split("/")[-1].split('.')[0]
        self.descr = np.empty(0)
        self.descr_path = descr_path
        self.nb_descr = DEFAULT_N_FEATURES
        self.group_id = int(self.name[:4])
        self.id = int(self.name[4:])

    def __hash__(self) -> int:
        return hash(self.path)


    def __eq__(self, other) -> bool:
        if not isinstance(other, Image) : 
            return NotImplemented
        return self.path == other.path and self.nb_descr == other.nb_descr
    
    def __lt__(self, other) : 
        if not isinstance(other, Image) : 
            return NotImplemented
        return self.path < other.path and self.nb_descr < other.nb_descr
    
    def has_same_group(self, other):
        if not isinstance(other, Image) : 
            return NotImplemented
        return self.group_id == other.group_id

    def load_descr(self, buffer_size=256):
        if self.descr_path == None:
            print("Cannot load, no descr_path provided")
            raise LoadError
        else:
            with open(self.descr_path, "rb") as file:

                lg = file.read(4)
                self.nb_descr = struct.unpack("<l", lg)[0]
                # nombre de descripteurs de l'image

                data = np.empty((self.nb_descr,128))
                i = 0
                for i in range(0, self.nb_descr): 
                    chunk = file.read(4 * 128)
                    descriptor = struct.unpack("<" + ("f" * 128), chunk)
                    data[i] = descriptor
                self.descr = data
    
    def compute_descr(self, save = False,
        outfile="", nfeatures=DEFAULT_N_FEATURES
    ):
        outfile = os.path.abspath(outfile)

        img = cv.imread(self.path)

        grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        sift = cv.SIFT_create(nfeatures=nfeatures) 

        _, des = sift.detectAndCompute(grayscale, None) 


        nbr_effectif_features = min(len(des), nfeatures)

        self.descr = np.array(des[:nbr_effectif_features])

        self.nb_descr = nbr_effectif_features

        self.descr.reshape((self.nb_descr,128)) # risque de bugs dans ce coin
        if save:
            # format de sortie : n : nbr de descripteur : 4 bytes, entier signé, little endiean, suivit de 128 * n flottants, chacun sur 4 bytes
            with open(outfile, "wb") as outfile:
                outfile.write(struct.pack("<l", self.nb_descr))

                for d in des:
                    for f in d:
                        outfile.write(
                            struct.pack("<f", f)
                        )  # c'est très probablement des flottants 32 bits donc f est ok



class Database:
    def __init__(self, dir_path : str = "", auto_init = True, verbose = False, nb_descr_per_img=DEFAULT_N_FEATURES) -> None:
        self.dir_path = os.path.abspath(dir_path)
        self.images = np.empty(0)
        self.nb_descr_per_img = nb_descr_per_img # nombre avec lequel ça a été calculé, ça peut être moins si y'a pas assez de features
        self.name = dir_path.split('/')[-1]
        self.descr_path = self.dir_path + "/../" + f"_descr_{self.nb_descr_per_img}_" + self.name
        if auto_init : 
            self.auto_init(verbose=verbose)


    def load_images(self) : 
        assert self.dir_path != None
        files_paths = [
            self.dir_path + "/" + f
            for f in os.listdir(self.dir_path)
            if os.path.isfile(self.dir_path + "/" + f)
        ]
        self.images = np.empty(len(files_paths), dtype=Image)

        descr_dir_exist = os.path.isdir(self.descr_path)
        for i,f_path in enumerate(files_paths):
            self.images[i] = Image(f_path)
            if descr_dir_exist :
                self.images[i].descr_path = self.descr_path + '/' + self.images[i].name
      
    def load_descriptors(self, verbose=False):
        #assert self.images != np.empty(0)
        assert self.dir_path != None
        if verbose:
            it = tqdm(range(len(self.images)), desc="Chargement des descripteurs")
        else:
            it = range(len(self.images))
        for i in it:
            self.images[i].load_descr()

    def compute_descr(self, save : bool = False, verbose = False):
        #assert self.images != np.empty(0)
        if verbose:
            it = tqdm(self.images, desc="Calcul des descripteurs")
        else:
            it = self.images
        
        
        os.mkdir(self.descr_path)
        for im in it :
            outfile = self.descr_path + '/' + im.name
                
            im.compute_descr(outfile = outfile, save = save, nfeatures = self.nb_descr_per_img)

    def auto_init(self, verbose = False) :
        
        self.load_images()
        if os.path.isdir(self.descr_path) :
            self.load_descriptors(verbose=verbose)
        else :
            self.compute_descr(save=True, verbose=verbose)

    def iter_descr(self):
        for im in self.images :
            for d in im.descr : 
                yield (d,im)
    
    def taille_nuage(self):
        return (sum(x.nb_descr for x in self.images))

    def to_array(self):
        tot_nb_descr = (sum(x.nb_descr for x in self.images))
         
        arr= np.empty((tot_nb_descr, 128), dtype=np.float32)
        for i, (d,_) in enumerate(self.iter_descr()):
            arr[i] = d

        return arr
    
    def image_of_descr_index(self, ind) : # détermine l'image associée au descripteur indexé ind (dans le tableau généré par to_array)
        s = self.images[0].nb_descr
        i = 0
        while (s <= ind):
            s += self.images[i].nb_descr
            i+=1
        return self.images[i]

if __name__ == "__main__":

    #test01()
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
    
    if entree_type :
        im = Image(entree)
        im.compute_descr()
    else :
        d = Database(entree, auto_init=True, verbose= True, nb_descr_per_img=nfeatures)
        a = d.to_array()
