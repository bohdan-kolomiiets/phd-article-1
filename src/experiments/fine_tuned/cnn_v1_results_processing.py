import os
import sys
import json
import argparse
import numpy as np
from libemg.datasets import *
from typing import cast

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from libemg_3dc.utils.training_experiments import TrainingExperiments, NeuralNetworkFineTunedTrainigExperiment

agrs_parser = argparse.ArgumentParser(description="Compare two classification models")
agrs_parser.add_argument('--transfer_learning_strategy', type=str, required=False, default='finetune_with_fc_reset', help='pass "finetune_with_fc_reset" or "finetune_without_fc_reset"')


if __name__ == "__main__":

    args = agrs_parser.parse_args()

    path = f'libemg_3dc/prove_pretraining_helps/fine_tuned/cnn_v1_results(ready {args.transfer_learning_strategy}).json'
    print(f'Path: "{path}"')
    training_results = TrainingExperiments.load(path=path)

    experiments: list[NeuralNetworkFineTunedTrainigExperiment] = [
        cast(NeuralNetworkFineTunedTrainigExperiment, result) for result in training_results.data if isinstance(result, NeuralNetworkFineTunedTrainigExperiment)]
    
    print(f"Loaded {len(experiments)} experiments")

    statistics_by_subject = {}
    for subject_id in range(0, 22):
        subject_experiments = [experiment for experiment in experiments if experiment.subject_id == subject_id]
        subject_f1_scores = [experiment.test_result["f1_score"] for experiment in subject_experiments]
        statistics_by_subject[subject_id] = {
            "mean" : np.mean(subject_f1_scores),
            "std": np.std(subject_f1_scores)
        }
            


    f1_scores = [result.test_result["f1_score"] for result in experiments]
    f1_score_mean = np.mean(f1_scores) # ?
    f1_score_std = np.std(f1_scores) # ?
    print(f"Mean F1-score: {f1_score_mean}")
    print(f"Std F1-score: {f1_score_std}")

    print(f"Statistics by subject: \n {json.dumps(statistics_by_subject, indent=2)}")

    print('The end')