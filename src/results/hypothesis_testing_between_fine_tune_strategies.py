import os
import sys
import argparse
import json
import numpy as np
from scipy.stats import wilcoxon
from scipy.stats import ttest_rel
from libemg.datasets import *
from typing import cast

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from libemg_3dc.utils.training_experiments import TrainingExperiments, NeuralNetworkSingleSubjectTrainingExperiment, NeuralNetworkFineTunedTrainigExperiment


def stringify_list(without_fc_reset_f1_score_means):
    return ', '.join([f"{v:.3f}" for v in without_fc_reset_f1_score_means])

if __name__ == "__main__":

    print()
    print(f"Performing hypothesis testing for transfer learning strategies 'finetune_without_fc_reset' and 'finetune_with_fc_reset'.")
    print()

    without_fc_reset_experiments = TrainingExperiments.load(path='libemg_3dc/prove_pretraining_helps/fine_tuned/cnn_v1_results(ready finetune_without_fc_reset).json')
    without_fc_reset_experiments: list[NeuralNetworkFineTunedTrainigExperiment] = [
        cast(NeuralNetworkFineTunedTrainigExperiment, result) for result in without_fc_reset_experiments.data if isinstance(result, NeuralNetworkFineTunedTrainigExperiment)]
    
    with_fc_reset_experiments = TrainingExperiments.load(path=f'libemg_3dc/prove_pretraining_helps/fine_tuned/cnn_v1_results(ready finetune_with_fc_reset).json')
    with_fc_reset_experiments: list[NeuralNetworkFineTunedTrainigExperiment] = [
        cast(NeuralNetworkFineTunedTrainigExperiment, result) for result in with_fc_reset_experiments.data if isinstance(result, NeuralNetworkFineTunedTrainigExperiment)]

    without_fc_reset_f1_scores = [result.test_result["f1_score"] for result in without_fc_reset_experiments]
    without_fc_reset_f1_score_mean = np.mean(without_fc_reset_f1_scores) 
    without_fc_reset_f1_score_std = np.std(without_fc_reset_f1_scores)
    print(f"without_fc_reset model's mean F1-score: {without_fc_reset_f1_score_mean:.3f}")
    print(f"without_fc_reset model's STD F1-score: {without_fc_reset_f1_score_std:.3f}")

    with_fc_reset_f1_scores = [result.test_result["f1_score"] for result in with_fc_reset_experiments]
    with_fc_reset_f1_score_mean = np.mean(with_fc_reset_f1_scores) 
    with_fc_reset_f1_score_std = np.std(with_fc_reset_f1_scores)
    print(f"with_fc_reset model's mean F1-score: {with_fc_reset_f1_score_mean:.3f}")
    print(f"with_fc_reset model's STD F1-score: {with_fc_reset_f1_score_std:.3f}")
    print()

    statistics_by_subject = []
    for subject_id in range(0, 22):
        subject_without_fc_reset_experiments = [experiment for experiment in without_fc_reset_experiments if experiment.subject_id == subject_id]
        subject_without_fc_reset_f1_scores = [experiment.test_result["f1_score"] for experiment in subject_without_fc_reset_experiments]
        
        subject_with_fc_reset_experiments = [experiment for experiment in with_fc_reset_experiments if experiment.subject_id == subject_id]
        subject_with_fc_reset_f1_scores = [experiment.test_result["f1_score"] for experiment in subject_with_fc_reset_experiments]
        
        statistics_by_subject.append({
            "without_fc_reset": {
                "mean" : np.mean(subject_without_fc_reset_f1_scores),
                "std": np.std(subject_without_fc_reset_f1_scores)
            },
            "with_fc_reset": {
                "mean" : np.mean(subject_with_fc_reset_f1_scores),
                "std": np.std(subject_with_fc_reset_f1_scores)
            }
        })

    without_fc_reset_f1_score_means = np.array([statistics["without_fc_reset"]["mean"] for statistics in statistics_by_subject])
    without_fc_reset_f1_score_stds = np.array([statistics["without_fc_reset"]["std"] for statistics in statistics_by_subject])

    with_fc_reset_f1_score_means = np.array([statistics["with_fc_reset"]["mean"] for statistics in statistics_by_subject])
    with_fc_reset_f1_score_stds = np.array([statistics["with_fc_reset"]["std"] for statistics in statistics_by_subject])

    print(f"without_fc_reset model's accuracy means: {stringify_list(without_fc_reset_f1_score_means)}")
    print(f"with_fc_reset model's accuracy means: {stringify_list(with_fc_reset_f1_score_means)}")
    print(f"Differences of models means (with_fc_reset minus without_fc_reset): {stringify_list(with_fc_reset_f1_score_means - without_fc_reset_f1_score_means)}")

    print(f"without_fc_reset model's accuracy STDs: {stringify_list(without_fc_reset_f1_score_stds)}")
    print(f"with_fc_reset model's accuracy STDs: {stringify_list(with_fc_reset_f1_score_stds)}")
    print(f"Differences of models STDs (without_fc_reset minus with_fc_reset): {stringify_list(without_fc_reset_f1_score_stds - with_fc_reset_f1_score_stds)}")

    print()
    print("Hypothesis testing:")
    threshold_p_value = 0.05

    print()
    print("Hypothesis testing for means:")
    print("Null hypothesis: with_fc_reset model gives the same accuracy as without_fc_reset model.")
    print("Alternative hypothesis: with_fc_reset model gives higher accuracy than without_fc_reset model.")
    
    print("Paried T-test:")
    t_mean_stat, t_mean_p_value = ttest_rel(with_fc_reset_f1_score_means, without_fc_reset_f1_score_means, alternative='greater')
    t_mean_conclusion = f"p-value: {t_mean_p_value:.3}<{threshold_p_value}. Null-hypothesis rejected. with_fc_reset model F1-score means are statistically greater than without_fc_reset model F1-score means. with_fc_reset model has higher accuracy." if t_mean_p_value < threshold_p_value else f"p-value: {t_mean_p_value:.3}>={threshold_p_value}. Cannot reject null-hypothesis. with_fc_reset model F1-score means cannot be proved to be statistically greater than without_fc_reset model F1-score means. with_fc_reset model is not proved to have higher accuracy."
    print(t_mean_conclusion)

    print("Wilcoxon signed-rank test:")
    w_mean_stat, w_mean_p_value = wilcoxon(with_fc_reset_f1_score_means, without_fc_reset_f1_score_means, alternative='greater')
    w_mean_conclusion = f"p-value: {w_mean_p_value:.3}<{threshold_p_value}. Null-hypothesis rejected. with_fc_reset model F1-score means are statistically greater than without_fc_reset model F1-score means. with_fc_reset model has higher accuracy." if w_mean_p_value < threshold_p_value else f"p-value: {w_mean_p_value:.3}>={threshold_p_value}. Cannot reject null-hypothesis. with_fc_reset model F1-score means cannot be proved to be statistically greater than without_fc_reset model F1-score means. with_fc_reset model is not proved to have higher accuracy."
    print(w_mean_conclusion)


    print()
    print("Hypothesis testing for STDs:")
    print("Null hypothesis: with_fc_reset model has as much stable stable accuracies as without_fc_reset model.")
    print("Alternative hypothesis: with_fc_reset model has more stable accuracies than without_fc_reset model.")

    print("Paried T-test:")
    t_std_stat, t_std_p_value = ttest_rel(with_fc_reset_f1_score_stds, without_fc_reset_f1_score_stds, alternative='less') 
    t_std_conclusion = f"p-value: {t_std_p_value:.3}<{threshold_p_value}. Null-hypothesis rejected. with_fc_reset model F1-score STDs are statistically less than without_fc_reset model F1-score STDs. with_fc_reset model is more stable." if t_std_p_value < threshold_p_value else f"p-value: {t_std_p_value:.3}>={threshold_p_value}. Cannot reject null-hypothesis. with_fc_reset model F1-score STDs cannot be proved to be statistically less than without_fc_reset model F1-score STDs. with_fc_reset model is not proved to be more stable."
    print(t_std_conclusion)

    print("Wilcoxon signed-rank test:")
    w_std_stat, w_std_p_value = wilcoxon(with_fc_reset_f1_score_stds, without_fc_reset_f1_score_stds, alternative='less')
    w_std_conclusion = f"p-value: {w_std_p_value:.3}<{threshold_p_value}. Null-hypothesis rejected. with_fc_reset model F1-score STDs are statistically less than without_fc_reset model F1-score STDs. with_fc_reset model is more stable." if w_std_p_value < threshold_p_value else f"p-value: {w_std_p_value:.3}>={threshold_p_value}. Cannot reject null-hypothesis. with_fc_reset model F1-score STDs cannot be proved to be statistically less than without_fc_reset model F1-score STDs. with_fc_reset model is not proved to be more stable."
    print(w_std_conclusion)

    print('The end.')