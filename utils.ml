let float_of_bytes buffer =
  let b' = Bytes.get_int32_le buffer 0 in
  Int32.float_of_bits b'

let int_of_bytes buffer = Int32.to_int (Bytes.get_int32_le buffer 0)

let get_4_bytes in_chan buffer =
  for i = 0 to 3 do
    let b = input_char in_chan in
    Bytes.set buffer i b
  done

let load_descriptors filepath =
  let cwd = Sys.getcwd () in
  let b = Str.string_match (Str.regexp "(.*\/)[A-Za-z0-9]") filepath 0 in
  if not b then failwith "Bad filepath";
  let dir = Str.matched_group 1 filepath in
  Sys.chdir dir;
  let file = open_in_bin filepath in
  try
    let buffer = Bytes.make 4 '\x00' in
    get_4_bytes file buffer;
    let size = int_of_bytes buffer in
    let data =
      Array.init size (fun i ->
          Array.init 128 (fun j ->
              get_4_bytes file buffer;
              float_of_bytes buffer))
    in
    Sys.chdir cwd;
    data
  with End_of_file -> failwith "Oups"

let load_data_descr dirpath =
  let cwd = Sys.getcwd () in
  Sys.chdir dirpath;
  let files = Sys.readdir dirpath in
  let n = Array.length files in
  let reg = Str.regexp "_descr.*" in
  let arr =
    Array.init n (fun i ->
        let f = files.(i) in
        if Str.string_match reg f 0 then load_descriptors f else [||])
  in
  Sys.chdir cwd;
  arr

let dist point_a point_b =
  let d = ref 0. in
  for i = 0 to 127 do
    let coord_dist =
      (point_a.(i) -. point_b.(i)) *. (point_a.(i) -. point_b.(i))
    in
    d := !d +. coord_dist
  done;
  sqrt !d
