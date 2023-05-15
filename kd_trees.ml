type point

type 'a kd_tree =
  | Empty_Leaf
  | Leaf of 'a
  | Node of 'a kd_tree * 'a kd_tree * int * 'a
(* Left subtree, right subtree, spliting dimension, spliting point*)

let build_tree_from_array (k : int) (points : 'a array) : 'a kd_tree =
  Array.sort compare points;
  (* Presort the points*)
  let rec aux axis st ed =
    (*start is inclusive, end is exclusive*)
    if st >= ed then Empty_Leaf
    else
      let mid = (st + ed) / 2 in
      let mid_elt = points.(mid) in
      let next_axis = axis + (1 mod k) in
      Node (aux next_axis st mid, aux next_axis (mid + 1) ed, axis, mid_elt)
  in
  aux 0 0 (Array.length points)
