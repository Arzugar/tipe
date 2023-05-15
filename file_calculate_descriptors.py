import cv2 as cv
import os
import sys
import struct
from tqdm import tqdm

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
                    )  # c'est très probablement des fottants 32 bits donc f est ok
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
