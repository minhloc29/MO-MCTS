import numpy as np
import time

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    """
    Design a novel algorithm to select the next node in each step.

    Args:
        current_node: ID of the current node.
        destination_node: ID of the destination node.
        unvisited_nodes: Array of IDs of unvisited nodes.
        distance_matrix: Distance matrix of nodes.

    Returns:
        ID of the next node to visit.
    """

    if len(unvisited_nodes) == 0:
        next_node = destination_node
    else:
        min_combined_cost = np.inf
        next_node = -1

        for candidate_node in unvisited_nodes:
            # Part 1: Calculate the immediate cost of traveling from the current node to the candidate node.
            cost_current_to_candidate = distance_matrix[current_node, candidate_node]

            # Part 2: Calculate the minimum cost to travel from the candidate node to any other remaining unvisited node.
            # This represents a greedy one-step lookahead from the candidate node.
            other_unvisited_nodes_excluding_candidate = [node for node in unvisited_nodes if node != candidate_node]

            if not other_unvisited_nodes_excluding_candidate:
                # If this candidate is the last unvisited node, there are no "other" unvisited nodes
                # to consider for a future minimum cost, so this component is zero.
                cost_min_to_next_unvisited = 0.0
            else:
                cost_min_to_next_unvisited = np.inf
                for other_node in other_unvisited_nodes_excluding_candidate:
                    cost_min_to_next_unvisited = min(cost_min_to_next_unvisited, distance_matrix[candidate_node, other_node])

            # Combine the two cost components with a different parameter setting:
            # Prioritize the one-step lookahead cost more heavily (e.g., by multiplying it by 2).
            combined_cost = cost_current_to_candidate + (2 * cost_min_to_next_unvisited)

            if combined_cost < min_combined_cost:
                min_combined_cost = combined_cost
                next_node = candidate_node

    return next_node

# Output variables: next_node
