import os
import sys
import numpy as np
import json
from libemg.datasets import *
from typing import cast

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from libemg_3dc.utils.training_experiments import TrainingExperiments, NeuralNetworkOtherSubjectsTrainingExperiment


if __name__ == "__main__":

    training_results = TrainingExperiments.load(path='libemg_3dc/prove_pretraining_helps/other_subjects/cnn_v1_results.json')

    experiments = [
        cast(NeuralNetworkOtherSubjectsTrainingExperiment, result) for result in training_results.data if isinstance(result, NeuralNetworkOtherSubjectsTrainingExperiment)]
    

    train_subjects_f1_scores = [experiment.training_subjects_test_result["f1_score"] for experiment in experiments]
    train_subjects_f1_score_mean = np.mean(train_subjects_f1_scores) 
    train_subjects_f1_score_std = np.std(train_subjects_f1_scores)
    print("train subjects:")
    print(f"mean F1-score: {train_subjects_f1_score_mean}")
    print(f"std F1-score: {train_subjects_f1_score_std}")

    statistics_by_subject = {}
    for subject_id in range(0, 22):
        subject_experiments = [experiment for experiment in experiments if experiment.test_subject_ids[0] == subject_id]
        subject_f1_scores = [experiment.training_subjects_test_result["f1_score"] for experiment in subject_experiments]
        statistics_by_subject[subject_id] = {
            "mean" : np.mean(subject_f1_scores),
            "std": np.std(subject_f1_scores)
        }
    print(f"Statistics by subject: \n {json.dumps(statistics_by_subject, indent=2)}")

    test_subjects_f1_scores = [experiment.test_subjects_test_result["f1_score"] for experiment in experiments]
    test_subjects_f1_score_mean = np.mean(test_subjects_f1_scores) 
    test_subjects_f1_score_std = np.std(test_subjects_f1_scores)
    print("test subjects:")
    print(f"mean F1-score: {test_subjects_f1_score_mean}")
    print(f"std F1-score: {test_subjects_f1_score_std}")

    statistics_by_subject = {}
    for subject_id in range(0, 22):
        subject_experiments = [experiment for experiment in experiments if experiment.test_subject_ids[0] == subject_id]
        subject_f1_scores = [experiment.test_subjects_test_result["f1_score"] for experiment in subject_experiments]
        statistics_by_subject[subject_id] = {
            "mean" : np.mean(subject_f1_scores),
            "std": np.std(subject_f1_scores)
        }
    print(f"Statistics by subject: \n {json.dumps(statistics_by_subject, indent=2)}")

    print('The end')