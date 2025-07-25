import math
from os import path
import numpy as np
import sys
import argparse
import time
from scipy.spatial import distance_matrix
import logging
from copy import copy

try:
    from gpt import select_next_node_v2 as select_next_node
except:
    from gpt import select_next_node


def eval_heuristic(node_positions: np.ndarray) -> float:
    '''
    Generate solution for TSP problem using the GPT-generated heuristic algorithm.
    
    Parameters
    ----------
    node_positions : np.ndarray
        2D array of node positions of shape (problem_size, 2).
    
    Returns
    -------
    obj : float
        The length of the generated tour.
    '''
    start_time = time.time()
    problem_size = node_positions.shape[0]
    # calculate distance matrix
    dist_mat = distance_matrix(node_positions, node_positions)
    # set the starting node
    start_node = 0
    solution = [start_node]
    # init unvisited nodes
    unvisited = np.array(list(range(problem_size)))
    # remove the starting node
    unvisited = unvisited[unvisited != start_node]

    # run the heuristic
    for _ in range(problem_size - 1):
        next_node = select_next_node(
            current_node=solution[-1],
            destination_node=start_node,
            unvisited_nodes=copy(unvisited),
            distance_matrix=dist_mat.copy(),
        )
        solution.append(next_node)
        if next_node in unvisited:
            unvisited = unvisited[unvisited != next_node]

        else:
            raise KeyError(f"Node {next_node} is already visited.")
    
    # calculate the length of the tour
    tour_length = 0
    for i in range(problem_size):
        tour_length += dist_mat[solution[i], solution[(i + 1) % problem_size]]
    
    end_time = time.time()
    runtime = end_time - start_time
    return tour_length, runtime
    

if __name__ == '__main__':
    print("[*] Running ...")

    problem_size = int(sys.argv[1])
    root_dir = sys.argv[2]
    mood = sys.argv[3]
    assert mood in ['train', 'val']

    basepath = path.join(path.dirname(__file__), "dataset")
    if not path.isfile(path.join(basepath, "train50_dataset.npy")):
        from gen_inst import generate_datasets
        generate_datasets()
    
    if mood == 'train':
        dataset_path = path.join(basepath, f"train{problem_size}_dataset.npy")
        node_positions = np.load(dataset_path)
        n_instances = node_positions.shape[0]
        print(f"[*] Dataset loaded: {dataset_path} with {n_instances} instances.")
        
        objs = []
        runtimes = []
        for i in range(n_instances):
            obj, runtime = eval_heuristic(node_positions[i])
            print(f"[*] Instance {i}: {obj}, {runtime}")
            objs.append(obj)
            runtimes.append(runtime)
        
        print(f"[*] Average Objectives:") # Objective 1
        print(np.mean(objs)) 
        
        print(f"[*] Average Runtime:") # Objective 2
        print(np.mean(runtimes))
    
    else:
        for problem_size in [20, 50, 100, 200]:
            dataset_path = path.join(basepath, f"val{problem_size}_dataset.npy")
            logging.info(f"[*] Evaluating {dataset_path}")
            node_positions = np.load(dataset_path)
            n_instances = node_positions.shape[0]
            objs = []
            runtimes = []
            for i in range(n_instances):
                obj, runtime = eval_heuristic(node_positions[i])
                objs.append(obj)
                runtimes.append(runtime)
                # print(f"[*] Average for {problem_size}: {obj}")
            print(f"[*] Average for {problem_size}: {np.mean(objs)}, {np.mean(runtimes)}")