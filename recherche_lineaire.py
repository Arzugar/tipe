import file_calculate_descriptors as dsu
import utils
import os
from tqdm import tqdm




#database = os.path.abspath (input())
print("Loading database")
database = "/home/lpottier/Documents/boulot/spe/TIPE/image_data/descr_very_small"
data = utils.load_database_descriptors(database, verbose=True)
print(f"Total number of points : {len(data)}")
print("Waiting for query")
#query = os.path.abspath (input())
query = "image_data/very_small/100002.jpg"
q_des = dsu.compute_descriptor_img(query,safe_path=True)

for i,point in tqdm(enumerate(q_des)) :
    m = min([utils.dist(point, data_point) for data_point in data ])
    print(f"Point {i} traité, distance minimale : {m}\n")
