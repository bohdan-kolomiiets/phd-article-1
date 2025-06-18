import os
import sys
import numpy as np
import json
from libemg.datasets import *
from typing import cast

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from libemg_3dc.utils.training_experiments import TrainingExperiments, NeuralNetworkSingleSubjectTrainingExperiment


if __name__ == "__main__":

    training_results = TrainingExperiments.load(path='libemg_3dc/prove_pretraining_helps/single_subject/cnn_v1_results(8 reps).json')

    experiments: list[NeuralNetworkSingleSubjectTrainingExperiment] = [
        cast(NeuralNetworkSingleSubjectTrainingExperiment, result) for result in training_results.data if isinstance(result, NeuralNetworkSingleSubjectTrainingExperiment)]
    
    statistics_by_subject = []
    for subject_id in range(0, 22):
        subject_experiments = [experiment for experiment in experiments if experiment.subject_id == subject_id]
        subject_f1_scores = [experiment.test_result["f1_score"] for experiment in subject_experiments]
        statistics_by_subject.append({
            "mean" : np.mean(subject_f1_scores),
            "std": np.std(subject_f1_scores)
        })


    f1_scores = [result.test_result["f1_score"] for result in experiments]
    f1_score_mean = np.mean(f1_scores) # 0.8685460978848195
    f1_score_std = np.std(f1_scores) # 0.08909870973418495
    print(f"mean F1-score: {f1_score_mean}")
    print(f"std F1-score: {f1_score_std}")

    print('The end')