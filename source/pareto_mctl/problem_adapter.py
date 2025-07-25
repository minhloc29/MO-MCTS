import logging
import os
import subprocess
import re
from typing import List, Any, Union, Dict, Any, Optional
import time
from utils.utils import block_until_running, file_to_string, filter_traceback
import math

class Prompts:
    def __init__(self, problem_cfg, root_dir: str):
        self.cfg = problem_cfg
        self.problem = problem_cfg.problem_name
        self.root_dir = root_dir
        self.problem_type = problem_cfg.problem_type
        self.prompt_dir = f"{self.root_dir}/prompts"

        prompt_path_suffix = "_black_box" if self.problem_type == "black_box" else ""
        problem_prompt_path = f'{self.prompt_dir}/{self.problem}{prompt_path_suffix}'
        self.func_signature = file_to_string(f'{problem_prompt_path}/func_signature.txt').format(version=2).strip()
        self.func_desc = file_to_string(f'{problem_prompt_path}/func_desc.txt')
        self.template_func = file_to_string(f'{problem_prompt_path}/template.txt')
        
        match = re.match(r'^def +(.+?)\((.*)\) *-> *(.*?) *:', self.func_signature)
        assert match is not None
        self.prompt_func_name = match.group(1)
        self.prompt_func_inputs = [txt.split(":")[0].strip() for txt in match.group(2).split(",")]
        if self.prompt_func_name.startswith('select_next_node'):
            self.prompt_func_outputs = ['next_node']
        elif self.prompt_func_name.startswith('priority'):
            self.prompt_func_outputs = ['priority']
        elif self.prompt_func_name.startswith('heuristics'):
            self.prompt_func_outputs = ['heuristics_matrix']
        elif self.prompt_func_name.startswith('crossover'):
            self.prompt_func_outputs = ['offsprings']
        elif self.prompt_func_name.startswith('utility'):
            self.prompt_func_outputs = ['utility_value']
        else:
            self.prompt_func_outputs = ['result']

    def get_task(self):
        return self.cfg.description

    def get_template(self):
        return self.template_func
    
    def get_func_name(self):
        return self.prompt_func_name

    def get_func_inputs(self):
        return self.prompt_func_inputs

    def get_func_outputs(self):
        return self.prompt_func_outputs

    def get_inout_inf(self):
        return self.func_desc

    def get_other_inf(self):
        return ""


class Problem:
    def __init__(self, cfg, root_dir):
        self.config = cfg
        self.root_dir = root_dir
        
        self.problem = self.config.problem.problem_name # ex: tsp_aco
        self.problem_description = self.config.problem.description # ex: Solving Traveling Salesman Problem (TSP) via stochastic solution sampling following "heuristics". TSP requires finding the shortest path that visits all given nodes and returns to the starting node.

        self.problem_size = self.config.problem.problem_size # ex: 50
        self.obj_types = [self.config.problem.obj_type, self.config.problem.obj_type] # min (or max)
        self.problem_type = self.config.problem.problem_type
        self.output_file = f"{self.root_dir}/problems/{self.problem}/gpt.py"
        self.num_objectives = self.config.problem.num_objectives # multi-objective
        if self.problem_type == "tsp_constructive":
            from .original.prompts.tsp_greedy import GetPrompts
            self.prompts = GetPrompts()
        elif self.problem_type == "bpp_online":
            from .original.prompts.bpp_online import GetPrompts
            self.prompts = GetPrompts()
        else:
            self.prompts = Prompts(self.config.problem, root_dir)

    def response_to_individual(self, code, response_id, file_name=None) -> dict:
        """
        Convert response to individual
        """
        outdir = './evaluations/'
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        runid = hash(code)
        # Write response to file
        file_name = outdir + f"problem_eval{runid}.txt" if file_name is None else file_name + ".txt"
        with open(file_name, 'w', encoding='utf-8') as file:
            file.writelines(code + '\n')

        # Extract code and description from response
        std_out_filepath = outdir + f"problem_eval{runid}_stdout.txt" if file_name is None else file_name.rstrip(
            ".txt") + "_stdout.txt"

        individual = {
            "stdout_filepath": std_out_filepath,
            "code_path": outdir + f"problem_eval{runid}_code.py",
            "code": code,
            "exec_success": False, 
            "response_id": response_id,
            "objective": [float('inf')] * self.num_objectives, 
        }
        return individual

    def mark_invalid_individual(self, individual: dict, traceback_msg: str) -> dict:
        """
        Mark an individual as invalid.
        """
        individual["exec_success"] = False
        individual["objective"] = [float("inf")] * self.num_objectives
        individual["traceback_msg"] = traceback_msg
        return individual

    def batch_evaluate(self, codes: List[str], iteration: int) -> Union[str, List[Dict[str, Any]]]: 
        """
        Evaluates a batch of heuristic codes in parallel.
        Measures individual execution time and parses all multi-objective values from stdout.
        
        Returns:
            'timeout' (str) if the entire batch communication times out (less likely with per-process timeout),
            or List[Dict[str, Any]] where each dict is an individual with its 'objective' (List[float]).
        """
        self.iteration = iteration
        population: List[Dict[str, Any]] = [self.response_to_individual(resp, index) for index, resp in enumerate(codes)]
        
        processes: List[Optional[subprocess.Popen]] = [] 
        launch_times: List[float] = [0.0] * len(population)

        for response_id in range(len(population)):
            individual = population[response_id]

            if individual["code"] is None:
                population[response_id] = self.mark_invalid_individual(individual, "Code is None!")
                processes.append(None)
                continue

            logging.info(f"Iteration {self.iteration}: Launching Code for response_id {response_id}")

            try:
                with open(self.output_file, 'w', encoding='utf-8') as file:
                    file.writelines(individual["code"] + '\n')
                
                launch_times[response_id] = time.time()
                eval_script_path = (
                    f'{self.root_dir}/problems/{self.problem}/eval.py' 
                    if self.problem_type != "black_box" 
                    else f'{self.root_dir}/problems/{self.problem}/eval_black_box.py'
                )
                
                process = subprocess.Popen(
                    ['python', '-u', eval_script_path, f'{self.problem_size}', self.root_dir, "train"], 
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                
                # --- TEMPORARY DEBUGGING CHANGE: Wait immediately for each process ---
                # Remove or comment out block_until_running here for testing if it's the culprit
                # block_until_running(individual["stdout_filepath"], log_status=True) 

                stdout_bytes, stderr_bytes = process.communicate(timeout=self.config.timeout)
                stdout_str = stdout_bytes.decode('utf-8', errors='ignore')
                stderr_str = stderr_bytes.decode('utf-8', errors='ignore')
                
                # Check exit code
                return_code = process.returncode
                print(f"Return code is: {return_code}")
                if return_code != 0:
                    logging.error(f"Process {response_id} exited with non-zero code {return_code}: {stderr_str}")
                else:
                    logging.info(f"Process {response_id} completed successfully.")
                    # Log captured stdout/stderr directly here to see content immediately
                    logging.debug(f"Process {response_id} STDOUT:\n{stdout_str}")
                    logging.debug(f"Process {response_id} STDERR:\n{stderr_str}")

                # --- Original logic for parsing and marking invalid now follows ---
                end_time = time.time()
                current_total_execution_time = end_time - launch_times[response_id]
                
                traceback_msg = filter_traceback(stderr_str if stderr_str else stdout_str) 

                if traceback_msg == '': 
                    try:
                        parsed_objective_values_from_stdout: List[float] = []
                        
                        stdout_lines = stdout_str.strip().split('\n')
                        logging.info(f"Stdout_lines: {stdout_lines}")
                        if len(stdout_lines) >= self.num_objectives:
                            for i in range(self.num_objectives):
                                try:
                                    val = float(stdout_lines[-i*2-1].strip())
                                    parsed_objective_values_from_stdout.append(val) # this is a list of objectives
                                except ValueError:
                                    parsed_objective_values_from_stdout = [float('inf')] * self.num_objectives
                                    logging.error(f"ValueError parsing objective {i} for response_id {response_id}. Output line: '{stdout_lines[-i]}'")
                                    break 
                            parsed_objective_values_from_stdout.reverse()

                            final_objectives: List[float] = []
                            logging.info('1')
                            if len(parsed_objective_values_from_stdout) == self.num_objectives and \
                               all(not math.isinf(o) and not math.isnan(o) for o in parsed_objective_values_from_stdout):
                                
                                for i, val in enumerate(parsed_objective_values_from_stdout):
                                    if i < len(self.obj_types) and self.obj_types[i] == "max":
                                        final_objectives.append(-val)
                                    else: 
                                        final_objectives.append(val)
                                if len(final_objectives) == self.num_objectives:
                                    individual["objective"] = final_objectives
                                    individual["exec_success"] = True
                                    logging.info("2")
                                else:
                                    logging.error(f"Parsed {len(final_objectives)} objectives, expected {self.num_objectives} for response_id {response_id}. Marking invalid.")
                                    individual["objective"] = [float('inf')] * self.num_objectives
                                    individual["exec_success"] = False

                            else:
                                individual["objective"] = [float('inf')] * self.num_objectives
                                individual["exec_success"] = False 

                        else: 
                             parsed_objective_values_from_stdout = [float('inf')] * self.num_objectives
                             logging.error(f"Not enough stdout lines ({len(stdout_lines)}) to parse {self.num_objectives} objectives for response_id {response_id}.")
                             individual["objective"] = parsed_objective_values_from_stdout
                             individual["exec_success"] = False

                    except Exception as e:
                        logging.error(f"Error processing objectives for response_id {response_id}: {e}", exc_info=True)
                        population[response_id] = self.mark_invalid_individual(individual, f"Invalid stdout/objective parse: {e}")
                else: 
                    population[response_id] = self.mark_invalid_individual(individual, traceback_msg)

                logging.info(f"Iteration {self.iteration}, response_id {response_id}: Objectives: {individual['objective']}")

            except Exception as e:
                logging.error(f"Error during individual processing for response_id {response_id}: {e}", exc_info=True)
                population[response_id] = self.mark_invalid_individual(individual, f"Outer loop error: {str(e)}")

        return population
        '''
        population = list of individuals
         individual = {
            "stdout_filepath": std_out_filepath,
            "code_path": outdir + f"problem_eval{runid}_code.py",
            "code": code,
            "exec_success": False, 
            "response_id": response_id,
            "objective": [float('inf')] * self.num_objectives, 
        }
        '''