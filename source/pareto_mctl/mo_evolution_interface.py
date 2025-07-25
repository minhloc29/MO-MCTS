import copy
import random
from typing import List, Tuple, Dict, Any
import numpy as np
import time
import logging
from ..common.evolution import Evolution
import warnings
from joblib import Parallel, delayed
import re
import concurrent.futures


class InterfaceEC():
    def __init__(self, m, api_endpoint, api_key, llm_model, debug_mode, interface_prob, select, n_p, timeout, use_numba,
                 **kwargs):
        assert 'use_local_llm' in kwargs
        assert 'url' in kwargs
        # -----------------------------------------------------------

        # LLM settings
        self.interface_eval = interface_prob
        prompts = interface_prob.prompts
        self.evol = Evolution(api_endpoint, api_key, llm_model, debug_mode, prompts, **kwargs)
        self.m = m
        self.debug = debug_mode

        if not self.debug:
            warnings.filterwarnings("ignore")

        self.select = select
        self.n_p = n_p # # Number of parallel processes

        self.timeout = timeout
        self.use_numba = use_numba

    def code2file(self, code):
        with open("./ael_alg.py", "w") as file:
            # Write the code to the file
            file.write(code)
        return

    def add2pop(self, population: List[Dict[str, Any]], offspring: Dict[str, Any]):
        for ind in population:
            if ind['objective'] == offspring['objective']: # check 2 lists
                if self.debug:
                    print("duplicated result, retrying ... ")
                return False
        population.append(offspring)
        return True

    def check_duplicate_obj(self, population: List[Dict[str, Any]], obj: List[float]):
        for ind in population:
            if obj == ind['objective']:
                return True
        return False

    def check_duplicate(self, population, code):
        for ind in population:
            if code == ind['code']:
                return True
        return False

    def population_generation_seed(self, seeds):

        population = []
        #eval_results: (fitness, runtime)
        eval_results = self.interface_eval.batch_evaluate([seed['code'] for seed in seeds])
        for i in range(len(seeds)):
            try:
                seed_alg = {
                    'algorithm': seeds[i]['algorithm'],
                    'code': seeds[i]['code'],
                    'objective': None, # List[float]
                    'other_inf': None
                }

                objective, runtime = eval_results[i]
                seed_alg['objective'] = np.round(objective, 5)
                seed_alg['runtime'] = runtime
                population.append(seed_alg)

            except Exception as e:
                print("Error in seed algorithm")
                exit()

        print("Initiliazation finished! Get " + str(len(seeds)) + " seed algorithms")

        return population

    def _get_alg(self, pop: List[Dict[str, Any]], operator, father=None):
        offspring = {
            'algorithm': None,
            'thought': None,
            'code': None,
            'objective': None, # List[float]
            'other_inf': None
        }
        if operator == "i1":
            parents = None
            [offspring['code'], offspring['thought']] = self.evol.i1()
        elif operator == "e1":
            real_m = random.randint(2, self.m)
            real_m = min(real_m, len(pop))
            parents = self.select.parent_selection_e1(pop, real_m)
            [offspring['code'], offspring['thought']] = self.evol.e1(parents)
        elif operator == "e2":
            other = copy.deepcopy(pop)
            if father in pop:
                other.remove(father)
            real_m = 1
            # real_m = random.randint(2, self.m) - 1
            # real_m = min(real_m, len(other))
            parents = self.select.parent_selection(other, real_m)
            parents.append(father)
            [offspring['code'], offspring['thought']] = self.evol.e2(parents)
        elif operator == "m1":
            parents = [father]
            [offspring['code'], offspring['thought']] = self.evol.m1(parents[0])
        elif operator == "m2":
            parents = [father]
            [offspring['code'], offspring['thought']] = self.evol.m2(parents[0])
        elif operator == "s1":
            parents = pop
            [offspring['code'], offspring['thought']] = self.evol.s1(pop)
        else:
            print(f"Evolution operator [{operator}] has not been implemented ! \n")

        offspring['algorithm'] = self.evol.post_thought(offspring['code'], offspring['thought'])
        return parents, offspring

    def get_offspring(self, pop, operator, father=None):
        while True:
            try:
                p, offspring = self._get_alg(pop, operator, father=father) # this cause error
                code = offspring['code']
                n_retry = 1
                while self.check_duplicate(pop, offspring['code']):
                    n_retry += 1
                    if self.debug:
                        print("duplicated code, wait 1 second and retrying ... ")
                    p, offspring = self._get_alg(pop, operator, father=father)
                    code = offspring['code']
                    if n_retry > 1:
                        break
                break
            except Exception as e:
                print(e)
        return p, offspring

    def get_algorithm(self, eval_times: int, pop: List[Dict[str, Any]], operator):
        '''
            operator: e1, e2, m1, m2, s1
        '''
        while True:
            eval_times += 1
            parents, offspring = self.get_offspring(pop, operator) # this cause the error
            eval_results = self.interface_eval.batch_evaluate([offspring['code']], 0)
            '''
                eval_results = population = list of individual
                 individual = {
                    "stdout_filepath": std_out_filepath,
                    "code_path": outdir + f"problem_eval{runid}_code.py",
                    "code": code,
                    "exec_success": False, 
                    "response_id": response_id,
                    "objective": [float('inf')] * self.num_objectives, 
                }
            '''
            evaluated_offspring = eval_results[0]
            offspring_objective_values: List[float] = evaluated_offspring['objective']
           
            if not evaluated_offspring.get('exec_success', False):
                if self.debug: print(f"Offspring execution failed: {evaluated_offspring.get('traceevck_msg', 'Unknown error')}")
                continue # Retry generation
            
            if self.check_duplicate_obj(pop, offspring_objective_values): # Pass the List[float]
                if self.debug: print(f"Offspring objective {offspring_objective_values} is dominated by or identical to existing, retrying.")
                continue # Retry generation

            offspring['objective'] = offspring_objective_values
            offspring['exec_success'] = True # Mark as successfully evaluated
            return eval_times, pop, offspring
             

    def evolve_algorithm(self, eval_times, pop, node, brother_node, operator):
        for i in range(3):
            eval_times += 1
            _, offspring = self.get_offspring(pop, operator, father=node)
            eval_results = self.interface_eval.batch_evaluate([offspring['code']], 0)
            '''
                eval_results = population = list of individual
                 individual = {
                    "stdout_filepath": std_out_filepath,
                    "code_path": outdir + f"problem_eval{runid}_code.py",
                    "code": code,
                    "exec_success": False, 
                    "response_id": response_id,
                    "objective": [float('inf')] * self.num_objectives, 
                }
            '''
            if eval_results == 'timeout':
                return eval_times, None
            
            evaluated_offspring = eval_results[0]
            offspring_objective_values: List[float] = evaluated_offspring['objective']
           
            if not evaluated_offspring.get('exec_success', False):
                if self.debug: print(f"Offspring execution failed: {evaluated_offspring.get('traceback_msg', 'Unknown error')}")
                continue # Retry generation
            
            if self.check_duplicate_obj(pop, offspring_objective_values): # Pass the List[float]
                if self.debug: print(f"Offspring objective {offspring_objective_values} is dominated by or identical to existing, retrying.")
                continue # Retry generation

            offspring['objective'] = offspring_objective_values
            offspring['exec_success'] = True # Mark as successfully evaluated

            return eval_times, offspring
        return eval_times, None
    
    @staticmethod
    def dominates(obj_a: List[float], obj_b: List[float]) -> bool:
        """
        Checks if obj_a (MINIMIZED values) dominates obj_b.
        obj_a dominates obj_b if obj_a is better than or equal to obj_b on all objectives,
        AND strictly better on at least one objective.
        """
        is_strictly_better_on_at_least_one = False
        for i in range(len(obj_a)):
            if obj_a[i] > obj_b[i]: # obj_a is worse on this objective
                return False # obj_a does not dominate obj_b
            if obj_a[i] < obj_b[i]: # obj_a is strictly better on this objective
                is_strictly_better_on_at_least_one = True
        return is_strictly_better_on_at_least_one
