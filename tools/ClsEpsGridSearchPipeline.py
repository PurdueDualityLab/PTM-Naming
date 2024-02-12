import numpy as np
from vector.ClusterPipeline import ClusterPipeline

class ClsEpsGridSearchPipeline():

    def __init__(self):
        self.cp = ClusterPipeline()

    def grid_search(
        self,
        vec_tuple: tuple,
        eval_func: callable,
        eps_list: list = np.linspace(0.1, 0.9, 9)
    ):
        results = []
        for eps in eps_list:
            result, outlier = self.cp.cluster_single_arch_from_dict(vec_tuple, eps=eps)
            results.append([eps, eval_func(result, outlier)])
        return max(results, key=lambda x: x[1])