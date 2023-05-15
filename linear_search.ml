Printexc.record_backtrace true;

Printf.printf "Loading database\n"

let database =
  "/home/lpottier/Documents/boulot/spe/TIPE/image_data/descr_very_small/"

let _ = Sys.chdir database
let data = Utils.load_data_descr database

let _ =
  Printf.printf "Total number of points : %d \n" (Array.length data);
  Printf.printf "Waiting for query\n"

let query_dir =
  "/home/lpottier/Documents/boulot/spe/TIPE/image_data/descr_very_small/"

let query_file = "_descr100000"
let _ = Sys.chdir query_dir
let q_des = Utils.load_descriptors query_file
let n = Array.length q_des
let p = Array.length data

let _ =
  Printf.printf "calculating\n";
  for i = 0 to n - 1 do
    for j = 0 to p - 1 do
      let d = Utils.dist q_des.(i) data.(j) in
      Printf.printf "%f\n" d
    done
  done
