"""
This module contains the tools for clustering the vectors.
"""

import os
from typing import List, Dict, Callable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import dotenv
from tqdm import tqdm
from sklearn.metrics import silhouette_score
import numpy as np
from vector.ClusterPipeline import ClusterPipeline


class GridSearchPipeline():
    """
    This class contains the tools for clustering the vectors.

    Attributes:
        model_embeddings (Dict[str, List[float]]): The embeddings of the input texts.
        cp (ClusterPipeline): The cluster pipeline.
    """

    def __init__(
        self,
        model_list: List[str],
    ):
        self.model_embeddings = self.get_embeddings(model_list)
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
            model="text-embedding-3-small"
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

        embeddings = [self.model_embeddings[model] for model in model_names]

        x = np.array(embeddings)
        y = np.array(labels, dtype=int)

        silhouette_avg = silhouette_score(x, y)
        return float(silhouette_avg)

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
        return {round(trial_list[i], rounding): round(results[i], rounding) for i in range(len(trial_list))}
