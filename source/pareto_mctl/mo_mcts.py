from __future__ import annotations
import random
import copy
import math
import numpy as np
from typing import List, Tuple, Any, Optional # Ensure Optional is imported
    
class MCTSNode:
    def __init__(self, algorithm, code, obj: List[float], depth=0, is_root=False, parent=None, visit=0, raw_info=None, Q_vector: List[float] = None):
        self.algorithm = algorithm
        self.code: str = code
        self.parent: MCTSNode = parent
        self.depth: int = depth
        self.children: List[MCTSNode] = [] # list of MCTSNode class
        self.children_info: List[dict] = [] # Raw info dictionaries of children, often used for prompting LLMs
        self.visits: int = visit
        self.subtree: List[MCTSNode] = []
        self.raw_info: List[MCTSNode] = raw_info
        self.Q_vector: List[float] = Q_vector if Q_vector is not None else [0.0] * len(obj)
        self.reward_vector: List[float] = -1 * np.array(obj) # a list, suppose all obj should be minimized so reward_vector should be maximize

    def add_child(self, child_node: MCTSNode):
        self.children.append(child_node)

    def __repr__(self):
        return f"MCTSNode(answer={self.algorithm}, Q={self.Q_vector}, visits={self.visits})"


class MCTS:
    def __init__(self, root_answer, num_objectives: int):
        self.exploration_constant_0 = 0.1 # Paramter \lambda_0
        self.alpha = 0.5 # Paramter \alpha
        self.max_depth = 10
        self.epsilon = 1e-10
        self.discount_factor = 1 # constant as 1
        self.q_min = [float('inf')] * num_objectives # Initialize q_min for each objective
        self.q_max = [float('-inf')] * num_objectives # Initialize q_max for each objective
        self.rank_list = []
        self.num_objectives = num_objectives
        self.root = MCTSNode(algorithm=root_answer, code=root_answer, depth=0, obj=[0.0] * num_objectives, is_root=True, Q_vector=[0.0] * num_objectives)
        self.global_pareto_front = [] # list of (reward vector of a node + node)
        # Logs
        self.critiques = []
        self.refinements = []
        self.rewards = []
        self.selected_nodes: List[MCTSNode] = []

    @staticmethod
    def dominates(reward_a: List[float], reward_b: List[float]) -> bool:
        is_strictly_better_on_at_least_one = False
        for i in range(len(reward_a)):
            if reward_a[i] < reward_b[i]: # If reward_a is worse on any objective
                return False # reward_a does not dominate reward_b
            if reward_a[i] > reward_b[i]: # If reward_a is strictly better on this objective
                is_strictly_better_on_at_least_one = True
        return is_strictly_better_on_at_least_one
    
    def update_global_pareto_front(self, new_solution_reward_vector: List[float], node_ref: MCTSNode):
        """
        Updates the global Pareto front with a new non-dominated solution.
        Assumes reward_vector elements are MAXIMIZED.
        """
        # Check if new_solution_reward_vector is dominated by any existing solution in the front
        is_dominated_by_existing = False
        for existing_reward_vec, _ in self.global_pareto_front:
            if self.dominates(existing_reward_vec, new_solution_reward_vector):
                is_dominated_by_existing = True
                break
        
        if not is_dominated_by_existing:
            # If new_solution_reward_vector is not dominated, add it and remove any solutions it dominates.
            updated_front = []
            for existing_reward_vec, existing_node_ref in self.global_pareto_front:
                if not self.dominates(new_solution_reward_vector, existing_reward_vec):
                    updated_front.append((existing_reward_vec, existing_node_ref))
            updated_front.append((new_solution_reward_vector, node_ref))
            self.global_pareto_front = updated_front

    def backpropagate(self, node: MCTSNode):
        # Update rank_list (assuming you want to rank based on some combination of objectives)
        # This part needs to be adapted based on how you want to rank multi-objective solutions
        # For example, you might use a scalarization function or dominance relation
        if node.Q_vector not in self.rank_list: 
            '''
                Q_vector: List[float]
                update parent visit of a node and parent.Q during backpropegate
            '''
            self.rank_list.append(node.Q_vector) # list of list
            self.rank_list.sort() # We sort in ascending order
        
        # Update q_min and q_max for each objective
        for i in range(self.num_objectives):
            self.q_min[i] = min(self.q_min[i], node.Q_vector[i])
            self.q_max[i] = max(self.q_max[i], node.Q_vector[i])

        parent = node.parent
        while parent: # how does parents update their Q
            # Update parent's Q_vector for each objective
            for i in range(self.num_objectives):
                parent.Q_vector[i] = parent.Q_vector[i] + (node.reward_vector[i] - parent.Q_vector[i]) / (parent.visits + 1) # incremental average
            
            parent.visits += 1
            if parent.code != 'Root' and parent.parent.code == 'Root':
                parent.subtree.append(node)
            parent = parent.parent

    def pareto_best_child(self, node: MCTSNode) -> MCTSNode: 
        """
        Selects the best child from the Pareto optimal set.
        input: a node, output: the best child (highest UCB)
        """
        # Compute Pareto UCB for each child
        pareto_ucb_values = []
        for child in node.children:
            ucb_vector = self.pareto_ucb(child, node.visits)
            pareto_ucb_values.append(ucb_vector)
        
        # Build approximate Pareto optimal node set based on UCB vectors
        pareto_optimal_set = self.build_pareto_front(node.children, pareto_ucb_values)
        
        # Choose a child from the Pareto optimal set uniformly at random
        if pareto_optimal_set:
            best_child = random.choice(pareto_optimal_set) # (among a list of non-dominated solutions, randomly choose one)
            return best_child
        else:
            return None  # Handle the case where the Pareto front is empty

    def pareto_ucb(self, node: MCTSNode, parent_visits: int) -> List[float]:
        """
        Computes the Pareto Upper Confidence Bound (UCB) for a node.
        """
        ucb_vector = []
        for i in range(self.num_objectives):
            exploration_term = math.sqrt((4 * math.log(parent_visits) + math.log(self.num_objectives)) / (2 * node.visits)) if node.visits > 0 else float('inf')
            ucb = node.Q_vector[i] + exploration_term
            ucb_vector.append(ucb)
        return ucb_vector

    def build_pareto_front(self, children: List[MCTSNode], ucb_values: List[List[float]]) -> List[MCTSNode]:
        """
        Builds an approximate Pareto optimal node set based on UCB vectors.
        Among a list of children, find the dominant ones, (list of non-dominated solutions)
        """
        pareto_front = []
        for i, child in enumerate(children):
            is_dominated = False
            for j, other_child in enumerate(children):
                if i == j:
                    continue
                if self.dominates(ucb_values[j], ucb_values[i]):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_front.append(child)
        return pareto_front

    