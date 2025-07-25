import heapq
from typing import List, Any, Dict, Tuple
def dominates(objective_a: List[float], objective_b: List[float]) -> bool:
    """
    Checks if objective_a dominates objective_b (all values are greater or equal, and at least one is strictly greater).
    Assumes minimalization.
    individual['objective'] is a List
    """
    is_strictly_better_on_at_least_one = False
    for i in range(len(objective_a)):
        if objective_a[i] > objective_b[i]:
            return False
        if objective_a[i] < objective_b[i]:
            is_strictly_better_on_at_least_one = True
    return is_strictly_better_on_at_least_one

def population_management(pop_input,size): # keep the population to number = size
    pop = [individual for individual in pop_input if individual['objective'] is not None]
    if size > len(pop):
        size = len(pop)
    unique_pop: List[Dict] = [] 
    unique_objectives: List[Tuple] = []
    for individual in pop:
        if individual['objective'] not in unique_objectives:
            objective = tuple(individual['objective'])
            if objective not in unique_objectives:
                unique_pop.append(individual)
                unique_objectives.append(objective)
    
    if len(unique_pop) > size:
        pareto_front = []
        remaining_pop = []

        for candidate in unique_pop:
            is_dominated = False
            for existing in pareto_front:
                if dominates(existing['objective'], candidate['objective']):
                    is_dominated = True
                    break
            if not is_dominated:
                # Remove dominated individuals from Pareto front
                pareto_front = [
                    existing
                    for existing in pareto_front
                    if not dominates(candidate['objective'], existing['objective'])
                ]
                pareto_front.append(candidate)
        
        # If Pareto front is smaller than size, fill with remaining population
        if len(pareto_front) < size:
            remaining_pop = [ind for ind in unique_pop if ind not in pareto_front]
            # Sort remaining population by hypervolume contribution (example - can be changed)
            remaining_pop = sorted(remaining_pop,
                                    key=lambda x: hypervolume_contribution(x['objective'], pareto_front),
                                    reverse=True)
            pareto_front.extend(remaining_pop[:size - len(pareto_front)])
        
        pop_new = pareto_front[:size] # Truncate to size
    else:
        pop_new = unique_pop[:size] # Truncate to size

    return pop_new

def population_management_s1(pop_input,size): # add tree path for reasoning s1
    pop = [individual for individual in pop_input if individual['objective'] is not None]
    if size > len(pop):
        size = len(pop)
    unique_pop = []
    unique_algorithms = []
    for individual in pop:
        if individual['algorithm'] not in unique_algorithms:
            unique_pop.append(individual)
            unique_algorithms.append(individual['algorithm'])

    pareto_front = []
    for candidate in unique_pop:
        is_dominated = False
        for existing in pareto_front:
            if dominates(existing['objective'], candidate['objective']):
                is_dominated = True
                break
        if not is_dominated:
            # Remove dominated individuals from Pareto front
            pareto_front = [
                existing
                for existing in pareto_front
                if not dominates(candidate['objective'], existing['objective'])
            ]
            pareto_front.append(candidate)
    
    # If Pareto front is smaller than size, fill with remaining population
    if len(pareto_front) < size:
        remaining_pop = [ind for ind in unique_pop if ind not in pareto_front]
        # Sort remaining population by hypervolume contribution (example - can be changed)
        remaining_pop = sorted(remaining_pop,
                                key=lambda x: hypervolume_contribution(x['objective'], pareto_front),
                                reverse=True)
        pareto_front.extend(remaining_pop[:size - len(pareto_front)])
    
    pop_new = pareto_front[:size] 
    
    return pop_new

def hypervolume_contribution(objective: List[float], pareto_front: List[Dict]) -> float:
    """
    Estimates the hypervolume contribution of an individual to the Pareto front.
    This is a simplified example and might need a more sophisticated implementation
    for higher dimensions.  Assumes maximization.
    """
    # Simple approximation: distance to the nearest neighbor in the Pareto front
    if not pareto_front:
        return 0.0  # No contribution if Pareto front is empty

    distances = [sum([(objective[i] - other['objective'][i])**2 for i in range(len(objective))])**0.5
                 for other in pareto_front]
    return min(distances)

