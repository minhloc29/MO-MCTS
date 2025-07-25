from source.mctl.mcts_ahd import MCTS_AHD
from source.common.getParas import Paras
from source.common import prob_rank
from source.mctl.single_problem_adapter import Problem

from source.mctl import pop_greedy
from utils.utils import init_client

class AHD:
    def __init__(self, cfg, root_dir, workdir, client) -> None:
        self.cfg = cfg
        self.root_dir = root_dir
        self.problem = Problem(cfg, root_dir)

        self.paras = Paras() 
        self.paras.set_paras(method = "mcts_ahd",
                             init_size = self.cfg.init_pop_size,
                             pop_size = self.cfg.pop_size,
                             llm_model = client,
                             ec_fe_max = self.cfg.max_fe,
                             exp_output_path = f"{workdir}/",
                             exp_debug_mode = False,
                             eva_timeout=cfg.timeout)
        init_client(self.cfg)
    
    def evolve(self):
        print("- Evolution Start -")

        method = MCTS_AHD(self.paras, self.problem, prob_rank, pop_greedy)

        results = method.run()

        print("> End of Evolution! ")
        print("-----------------------------------------")
        print("---  MCTS-AHD successfully finished!  ---")
        print("-----------------------------------------")

        return results


