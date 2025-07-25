import re
import time
from .interface_LLM import InterfaceAPI as InterfaceLLM
import re
from typing import List
import copy
from typing import List, Tuple, Dict, Any
import logging
from utils.utils import file_to_string
input = lambda: ...
import os
import textwrap
class Evolution():

    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode, prompts, **kwargs):
        assert 'url' in kwargs
        self._url = kwargs.get('url')
        # -----------------------------------------------------------
        # visit cfg/problem
        self.prompt_task = prompts.get_task()
        self.prompt_func_name = prompts.get_func_name()
        self.prompt_func_inputs = prompts.get_func_inputs()
        self.prompt_func_outputs = prompts.get_func_outputs()
        self.prompt_inout_inf = prompts.get_inout_inf()
        self.prompt_other_inf = prompts.get_other_inf()
        self.template_func = prompts.get_template()
        #Prompt Template
        self.init_prompt = file_to_string("/Users/macbook/Documents/Code/MCTS-AHD-master/prompt_strategy/init.txt")
        self.e1_prompt = file_to_string("/Users/macbook/Documents/Code/MCTS-AHD-master/prompt_strategy/e1.txt")
        self.e2_prompt = file_to_string("/Users/macbook/Documents/Code/MCTS-AHD-master/prompt_strategy/e2.txt")
        self.m1_prompt = file_to_string("/Users/macbook/Documents/Code/MCTS-AHD-master/prompt_strategy/m1.txt")
        self.m2_prompt = file_to_string("/Users/macbook/Documents/Code/MCTS-AHD-master/prompt_strategy/m2.txt")
        self.s1_prompt = file_to_string("/Users/macbook/Documents/Code/MCTS-AHD-master/prompt_strategy/s1.txt")
        self.post_prompt = file_to_string("/Users/macbook/Documents/Code/MCTS-AHD-master/prompt_strategy/prompt_post.txt")
        self.refine_prompt = file_to_string("/Users/macbook/Documents/Code/MCTS-AHD-master/prompt_strategy/prompt_refine.txt")

        if len(self.prompt_func_inputs) > 1:
            self.joined_inputs = ", ".join("'" + s + "'" for s in self.prompt_func_inputs)
        else:
            self.joined_inputs = "'" + self.prompt_func_inputs[0] + "'"

        if len(self.prompt_func_outputs) > 1:
            self.joined_outputs = ", ".join("'" + s + "'" for s in self.prompt_func_outputs)
        else:
            self.joined_outputs = "'" + self.prompt_func_outputs[0] + "'"

        # set LLMs
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode  # close prompt checking

        self.interface_llm = InterfaceLLM(self.api_endpoint, self.api_key, self.model_LLM, self.debug_mode)
    
    def get_prompt_refine(self, code: str, algorithm: str):

        prompt_content = self.prompt_task + "\n" + "Following is the Design Idea of a heuristic algorithm for the problem and the code with function name '" + self.prompt_func_name + "' for implementing the heuristic algorithm.\n"
        prompt_content += self.prompt_inout_inf + " " + self.prompt_other_inf
        prompt_content += "\nDesign Idea:\n" + algorithm
        prompt_content += "\n\nCode:\n" + code
        prompt_content += "\n\nThe content of the Design Idea idea cannot fully represent what the algorithm has done informative. So, now you should re-describe the algorithm using less than 3 sentences.\n"
        prompt_content += "Hint: You should reference the given Design Idea and highlight the most critical design ideas of the code. You can analyse the code to describe which variables are given higher priorities and which variables are given lower priorities, the parameters and the structure of the code."
        logging.info(f"Prompt Refine: {prompt_content}")
        return prompt_content

    def get_prompt_post(self, code: str, algorithm: str): #after input the prompt idea and algo, should output the detailed idea again

        prompt_content = self.prompt_task + "\n" + "Following is the a Code implementing a heuristic algorithm with function name " + self.prompt_func_name + " to solve the above mentioned problem.\n"
        prompt_content += self.prompt_inout_inf + " " + self.prompt_other_inf
        prompt_content += "\n\nCode:\n" + code
        prompt_content += "\n\nNow you should describe the Design Idea of the algorithm using less than 5 sentences.\n"
        prompt_content += "Hint: You should highlight every meaningful designs in the provided code and describe their ideas. You can analyse the code to see which variables are given higher values and which variables are given lower values, the choice of parameters or the total structure of the code."
        logging.info(f"Prompt Post: {prompt_content}")
        return prompt_content

    def _get_thought(self, prompt_content: str):

        response = self.interface_llm.get_response(prompt_content, 0)

        # algorithm = response.split(':')[-1]
        return response

    def clean_llm_code(self, raw_code: str) -> str:
        """
        Fully clean LLM code response: normalize tabs, remove empty leading lines,
        dedent, and strip trailing whitespace.
        """
        raw_code = raw_code.replace('\t', '    ')  # Convert tabs to spaces
        lines = raw_code.splitlines()

        # Remove leading empty lines
        while lines and lines[0].strip() == "":
            lines.pop(0)

        cleaned = textwrap.dedent('\n'.join(lines)).strip()
        return cleaned
    
    def _get_algo_base_prompt(self, prompt_content):
        logging.info("Calling LLM to generate algorithm and code...")

        max_retries = 3
        for attempt in range(max_retries):
            response = self.interface_llm.get_response(prompt_content)
            logging.debug(f"Raw LLM response (attempt {attempt}):\n{repr(response)}")

            # === Extract algorithm ===
            algorithm = ""
            try:
                match = re.search(r"\{(.*?)\}", response, re.DOTALL)
                if match:
                    algorithm = match.group(1).strip()
            except Exception as e:
                logging.warning(f"Algorithm extraction failed: {e}")

            # === Extract code block ===
            code_blocks = re.findall(r"```(?:python)?\s*([\s\S]*?)```", response)
            if not code_blocks:
                # Fallback to looser patterns
                code_blocks = re.findall(r"(import[\s\S]+?def[\s\S]+?return[\s\S]*?)", response)
            if not code_blocks:
                code_blocks = re.findall(r"(def\s+[^\n]*:[\s\S]+?return[^\n]*)", response)

            if code_blocks:
                raw_code = code_blocks[0]
                cleaned_code = self.clean_llm_code(raw_code)
                code_all = cleaned_code + "\n\n# Output variables: " + ", ".join(self.prompt_func_outputs)
                return [code_all, algorithm]

            logging.warning(f"Attempt {attempt + 1} failed. Retrying...")

        # After retries
        raise ValueError("❌ Failed to extract both algorithm and code from LLM response.")


    def post_thought(self, code: str, algorithm: str):

        prompt_content = self.get_prompt_refine(code, algorithm)

        post_thought = self._get_thought(prompt_content)

        return post_thought

    def i1(self):
        prompt_content = self.init_prompt.format(task_prompt = self.prompt_task, temp_func = self.template_func)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ i1 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_algo_base_prompt(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()
        logging.info(f"Prompt i1: {prompt_content}")
        return [code_all, algorithm]

    def e1(self, parents: List[Dict[str, Any]]):
        '''
            offspring = {
            'algorithm': None,
            'thought': None,
            'code': None,
            'objective': None, List[float]
            'other_inf': None
        }
        '''
        for indi in parents:
            assert ('algorithm' in indi), "Algorithm key is not included in individual dictionary"

        indivs_prompt = ''
        for i, indi in enumerate(parents):
            indivs_prompt += f'No. {i + 1} algorithm and the corresponding code are:\n{indi['algorithm']}\n{indi['code']}'
            
        prompt_content = self.e1_prompt.format(task_prompt = self.prompt_task, algo_length = len(parents), list_of_codes = indivs_prompt, temp_func = self.template_func)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e1 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()
        [code_all, algorithm] = self._get_algo_base_prompt(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        logging.info(f"Prompt e1: {prompt_content}")
        return [code_all, algorithm]

    def e2(self, parents: List[Dict[str, Any]]):
        for indi in parents:
            assert ('algorithm' in indi), "Algorithm key is not included in individual dictionary"
            
        indivs_prompt = ''
        for i, indi in enumerate(parents):
            indivs_prompt += f'No. {i + 1} algorithm and the corresponding code are:\n{indi['algorithm']}\n{indi['code']}'
            
        prompt_content = self.e2_prompt.format(task_prompt = self.prompt_task, algo_length = len(parents), list_of_codes = indivs_prompt, temp_func = self.template_func)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e2 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_algo_base_prompt(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        logging.info(f"Prompt e2: {prompt_content}")
        return [code_all, algorithm]

    def m1(self, parents: Dict[str, Any]):
        assert ('algorithm' in parents), "Algorithm key is not included in individual dictionary"
        prompt_content = self.m1_prompt.format(task_prompt = self.prompt_task, algo_desc = parents['algorithm'],
                                               code = parents['code'], temp_func = self.template_func)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m1 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_algo_base_prompt(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        logging.info(f"Prompt m1: {prompt_content}")
        return [code_all, algorithm]

    def m2(self, parents: Dict[str, Any]):

        assert ('algorithm' in parents), "Algorithm key is not included in individual dictionary"
        prompt_content = self.m2_prompt.format(task_prompt = self.prompt_task, algo_desc = parents['algorithm'],
                                               code = parents['code'], temp_func = self.template_func)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m2 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_algo_base_prompt(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        logging.info(f"Prompt m2: {prompt_content}")
        return [code_all, algorithm]

    def s1(self, parents: List[Dict[str, Any]]):

        for indi in parents:
            assert ('algorithm' in indi), "Algorithm key is not included in individual dictionary"

        indivs_prompt = ''
        for i, indi in enumerate(parents):
            indivs_prompt += f'No. {i + 1} algorithm and the corresponding code are:\n{indi['algorithm']}\n{indi['code']}'
            
        prompt_content = self.s1_prompt.format(task_prompt = self.prompt_task, algo_length = len(parents), list_of_codes = indivs_prompt, temp_func = self.template_func)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ s1 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_algo_base_prompt(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        logging.info(f"Prompt s1: {prompt_content}")
        return [code_all, algorithm]
