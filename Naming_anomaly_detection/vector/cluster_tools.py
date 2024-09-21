"""
This module contains the tools for clustering the vectors.
"""

import os
from typing import List, Dict, Callable, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import dotenv
from tqdm import tqdm
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
import numpy as np
from loguru import logger
from vector.cluster_pipeline import ClusterPipeline


class GridSearchPipeline():
    """
    This class contains the tools for clustering the vectors.

    Attributes:
        model_embeddings (Dict[str, List[float]]): The embeddings of the input texts.
        cp (ClusterPipeline): The cluster pipeline.
        
    """

    def __init__(
        self,
        model_list: Optional[List[str]] = None,
        model_embeddings: Optional[Dict[str, List[float]]] = None,
    ):
        if model_embeddings is None:
            if model_list is None:
                raise ValueError("Either model_list or model_embeddings should be provided.")
            self.model_embeddings = self.get_embeddings(model_list)
        else:
            self.model_embeddings = model_embeddings
        self.cp = ClusterPipeline()

    def get_embeddings(
            self,
            texts: List[str]
        ) -> Dict[str, List[float]]:
        """
        This function returns the embeddings of the input texts.

        Args:
            texts (List[str]): The input texts.
        
        Returns:
            Dict[str, List[float]]: A dictionary with the input 
            texts as keys and the embeddings as values.
        """
        dotenv.load_dotenv(".env", override=True)
        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(
            input=texts,
            model="text-embedding-3-small",
            #dimensions=64
        )
        return {texts[i]: response.data[i].embedding for i in range(len(texts))}

    def get_silhouette_score(
            self,
            result: Dict[str, Dict[str, List[str]]]
        ) -> float:
        """
        This function returns the silhouette score of the input result.

        Args:
            result (Dict[str, Dict[str, List[str]]]): The input result.

        Returns:
            float: The silhouette score of the input result.
        """
        model_names = []
        labels = []

        for groups in result.values():
            for label, models in groups.items():
                for model in models:
                    model_names.append(model)
                    labels.append(label)

        embeddings = [
            self.model_embeddings[list(result.keys())[0]][model_names[i]] \
                for i in range(len(model_names))
        ]

        x = np.array(embeddings)
        y = np.array(labels, dtype=int)

        if len(set(labels)) < 2:
            return -1.0

        silhouette_avg = silhouette_score(x, y)
        return float(silhouette_avg)
    
    def get_davies_bouldin_score(self, result: Dict[str, Dict[str, List[str]]]) -> float:
        """
        This function returns the Davies-Bouldin score of the input result.

        Args:
            result (Dict[str, Dict[str, List[str]]]): The input result.

        Returns:
            float: The Davies-Bouldin score of the input result. Lower DBI values indicate better clustering.
        """
        model_names = []
        labels = []

        for groups in result.values():
            for label, models in groups.items():
                for model in models:
                    model_names.append(model)
                    labels.append(label)

        embeddings = [
            self.model_embeddings[list(result.keys())[0]][model_names[i]]
                for i in range(len(model_names))
        ]

        x = np.array(embeddings)
        y = np.array(labels, dtype=int)

        # The DBI requires at least two clusters to be non-trivial, hence the check
        if len(set(labels)) < 2:
            # It is not possible to calculate DBI with less than 2 clusters
            # Returning some high value or an indication that it's not applicable
            return float("inf") 

        dbi = davies_bouldin_score(x, y)
        return float(dbi)
    
    def combination_metric(
        self,
        result: Dict[str, Dict[str, List[str]]]
    ) -> float:
        """
        This function returns the combination metric of the input result.

        Args:
            result (Dict[str, Dict[str, List[str]]]): The input result.

        Returns:
            float: The combination metric of the input result.
        """
        silhouette_score = self.get_silhouette_score(result)
        davies_bouldin_score = self.get_davies_bouldin_score(result)
        return silhouette_score - davies_bouldin_score

    def grid_search(
        self,
        vec_tuple: tuple,
        eval_func: Callable[..., Any],
        trial_list: np.ndarray,
        rounding: int = 14,
        parallel_processing: bool = True
    ):
        """
        This function returns the optimal value of the input result.

        Args:
            vec_tuple (tuple): The input vector tuple.
            eval_func (callable): The evaluation function.
            trial_list (List[float]): The trial list.
            rounding (int): The rounding value.
            parallel_processing (bool): The parallel processing flag.

        Returns:
            Dict[float, float]: A dictionary with the trial list as keys and 
            the evaluation results as values.
        """
        results = []

        def task(eps):
            result = self.cp.cluster_single_arch_from_dict(
                vec_tuple,
                eps=eps,
                merge_outlier=True
            )
            return eps, eval_func(result)

        if parallel_processing:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(task, eps) for eps in trial_list]
                results = []
                for future in tqdm(as_completed(futures), total=len(futures)):
                    eps, eval_result = future.result()
                    results.append((eps, eval_result))
        else:
            results = []
            for eps in tqdm(trial_list):
                eps, eval_result = task(eps)
                results.append((eps, eval_result))
        return {round(eps, rounding): round(eval_result, rounding) for eps, eval_result in results}

    def search_optimal_eps(
        self,
        vec_tuple: tuple,
        eval_func: Callable[..., Any],
        rounding: int = 14,
        parallel_processing: bool = True
    ):
        """
        This function returns the optimal value of the input result.

        Args:
            vec_tuple (tuple): The input vector tuple.
            eval_func (callable): The evaluation function.
            rounding (int): The rounding value.
            parallel_processing (bool): The parallel processing flag.
        
        Returns:
            None
        """
        span = 5.0
        mid = 0.0
        curr_trial_list = np.logspace(mid-span, mid+span, 10)
        for _ in range(5):
            result = self.grid_search(
                vec_tuple,
                eval_func,
                curr_trial_list,
                rounding,
                parallel_processing
            )
            largest_value_key = max(result, key=lambda k: float(result[k])) # pylint: disable=cell-var-from-loop
            log_largest_value_key = np.log10(largest_value_key)
            span = span / 2
            curr_trial_list = np.logspace(
                log_largest_value_key-span,
                log_largest_value_key+span,
                10
            )
            logger.info(
                f"Current largest value: {largest_value_key} with value {result[largest_value_key]}"
            )
            logger.info(f"Current range: {curr_trial_list[0]} to {curr_trial_list[-1]}")
        return largest_value_key
