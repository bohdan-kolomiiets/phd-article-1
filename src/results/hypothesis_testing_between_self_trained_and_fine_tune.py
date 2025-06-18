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


def stringify_list(self_training_f1_score_means):
    return ', '.join([f"{v:.3f}" for v in self_training_f1_score_means])


# agrs_parser = argparse.ArgumentParser(description="Compare two classification models")
# agrs_parser.add_argument('--transfer_learning_strategy', type=str, required=False, default='finetune_without_fc_reset', help='pass "finetune_with_fc_reset" or "finetune_without_fc_reset"')

if __name__ == "__main__":

    # args = agrs_parser.parse_args()
    # print()
    # print(f"Performing hypothesis testing for transfer learning strategy '{args.transfer_learning_strategy}'.")
    # print()

    single_subject_path = 'libemg_3dc/prove_pretraining_helps/single_subject/cnn_v1_results(3 reps).json'
    fine_tuned_path = f'libemg_3dc/prove_pretraining_helps/fine_tuned/cnn_v1_results(finetune_with_fc_reset 3 reps).json'
   
    single_subject_experiments = TrainingExperiments.load(path=single_subject_path)
    print(f"Loaded {len(single_subject_experiments.data)} experiments from '{single_subject_path}'")
    single_subject_experiments: list[NeuralNetworkSingleSubjectTrainingExperiment] = [
        cast(NeuralNetworkSingleSubjectTrainingExperiment, result) for result in single_subject_experiments.data if isinstance(result, NeuralNetworkSingleSubjectTrainingExperiment)]
    
    fine_tuned_experiments = TrainingExperiments.load(path=fine_tuned_path)
    print(f"Loaded {len(fine_tuned_experiments.data)} experiments from '{fine_tuned_path}'")
    fine_tuned_experiments: list[NeuralNetworkFineTunedTrainigExperiment] = [
        cast(NeuralNetworkFineTunedTrainigExperiment, result) for result in fine_tuned_experiments.data if isinstance(result, NeuralNetworkFineTunedTrainigExperiment)]

    single_subject_f1_scores = [result.test_result["f1_score"] for result in single_subject_experiments]
    single_subject_f1_score_mean = np.mean(single_subject_f1_scores) 
    single_subject_f1_score_std = np.std(single_subject_f1_scores)
    print(f"Self-trained model's mean F1-score: {single_subject_f1_score_mean:.3f}")
    print(f"Self-trained model's STD F1-score: {single_subject_f1_score_std:.3f}")

    fine_tuned_f1_scores = [result.test_result["f1_score"] for result in fine_tuned_experiments]
    fine_tuned_f1_score_mean = np.mean(fine_tuned_f1_scores) 
    fine_tuned_f1_score_std = np.std(fine_tuned_f1_scores)
    print(f"Fine-tuned model's mean F1-score: {fine_tuned_f1_score_mean:.3f}")
    print(f"Fine-tuned model's STD F1-score: {fine_tuned_f1_score_std:.3f}")
    print()

    statistics_by_subject = []
    for subject_id in range(0, 22):
        subject_self_training_experiments = [experiment for experiment in single_subject_experiments if experiment.subject_id == subject_id]
        subject_self_training_f1_scores = [experiment.test_result["f1_score"] for experiment in subject_self_training_experiments]
        
        subject_fine_tuning_experiments = [experiment for experiment in fine_tuned_experiments if experiment.subject_id == subject_id]
        subject_fine_tuning_f1_scores = [experiment.test_result["f1_score"] for experiment in subject_fine_tuning_experiments]
        
        statistics_by_subject.append({
            "single_subject": {
                "mean" : np.mean(subject_self_training_f1_scores),
                "std": np.std(subject_self_training_f1_scores)
            },
            "fine_tuned": {
                "mean" : np.mean(subject_fine_tuning_f1_scores),
                "std": np.std(subject_fine_tuning_f1_scores)
            }
        })

    self_training_f1_score_means = np.array([statistics["single_subject"]["mean"] for statistics in statistics_by_subject])
    self_training_f1_score_stds = np.array([statistics["single_subject"]["std"] for statistics in statistics_by_subject])

    fine_tuning_f1_score_means = np.array([statistics["fine_tuned"]["mean"] for statistics in statistics_by_subject])
    fine_tuning_f1_score_stds = np.array([statistics["fine_tuned"]["std"] for statistics in statistics_by_subject])

    print(f"Self-trained model's accuracy means: {stringify_list(self_training_f1_score_means)}")
    print(f"Fine-tuned model's accuracy means: {stringify_list(fine_tuning_f1_score_means)}")
    print(f"Differences of models means (fine-tuned minus self-trained): {stringify_list(fine_tuning_f1_score_means - self_training_f1_score_means)}")

    print(f"Self-trained accuracy STDs: {stringify_list(self_training_f1_score_stds)}")
    print(f"Fine-tuned accuracy STDs: {stringify_list(fine_tuning_f1_score_stds)}")
    print(f"Differences of models STDs (self-trained minus fine-tuned): {stringify_list(self_training_f1_score_stds - fine_tuning_f1_score_stds)}")

    print()
    print("Hypothesis testing:")
    threshold_p_value = 0.05

    print()
    print("Hypothesis testing for means:")
    print("Null hypothesis: Fine-tuned model gives the same accuracy as single-subject model.")
    print("Alternative hypothesis: Fine-tuned model gives higher accuracy than single-subject model.")
    
    print("Paried T-test:")
    t_mean_stat, t_mean_p_value = ttest_rel(fine_tuning_f1_score_means, self_training_f1_score_means, alternative='greater')
    t_mean_conclusion = f"p-value: {t_mean_p_value:.3}<{threshold_p_value}. Null-hypothesis rejected. Fine-tuned model F1-score means are statistically greater than subject-specific model F1-score means. Fine-tuned model has higher accuracy." if t_mean_p_value < threshold_p_value else f"p-value: {t_mean_p_value:.3}>={threshold_p_value}. Cannot reject null-hypothesis. Fine-tuned model F1-score means cannot be proved to be statistically greater than subject-specific model F1-score means. Fine-tuned model is not proved to have higher accuracy."
    print(t_mean_conclusion)

    print("Wilcoxon signed-rank test:")
    w_mean_stat, w_mean_p_value = wilcoxon(fine_tuning_f1_score_means, self_training_f1_score_means, alternative='greater')
    w_mean_conclusion = f"p-value: {w_mean_p_value:.3}<{threshold_p_value}. Null-hypothesis rejected. Fine-tuned model F1-score means are statistically greater than subject-specific model F1-score means. Fine-tuned model has higher accuracy." if w_mean_p_value < threshold_p_value else f"p-value: {w_mean_p_value:.3}>={threshold_p_value}. Cannot reject null-hypothesis. Fine-tuned model F1-score means cannot be proved to be statistically greater than subject-specific model F1-score means. Fine-tuned model is not proved to have higher accuracy."
    print(w_mean_conclusion)


    print()
    print("Hypothesis testing for STDs:")
    print("Null hypothesis: Fine-tuned model has as much stable stable accuracies as single-subject model.")
    print("Alternative hypothesis: Fine-tuned model has more stable accuracies than single-subject model.")

    print("Paried T-test:")
    t_std_stat, t_std_p_value = ttest_rel(fine_tuning_f1_score_stds, self_training_f1_score_stds, alternative='less') 
    t_std_conclusion = f"p-value: {t_std_p_value:.3}<{threshold_p_value}. Null-hypothesis rejected. Fine-tuned model F1-score STDs are statistically less than subject-specific model F1-score STDs. Fine-tuned model is more stable." if t_std_p_value < threshold_p_value else f"p-value: {t_std_p_value:.3}>={threshold_p_value}. Cannot reject null-hypothesis. Fine-tuned model F1-score STDs cannot be proved to be statistically less than subject-specific model F1-score STDs. Fine-tuned model is not proved to be more stable."
    print(t_std_conclusion)

    print("Wilcoxon signed-rank test:")
    w_std_stat, w_std_p_value = wilcoxon(fine_tuning_f1_score_stds, self_training_f1_score_stds, alternative='less')
    w_std_conclusion = f"p-value: {w_std_p_value:.3}<{threshold_p_value}. Null-hypothesis rejected. Fine-tuned model F1-score STDs are statistically less than subject-specific model F1-score STDs. Fine-tuned model is more stable." if w_std_p_value < threshold_p_value else f"p-value: {w_std_p_value:.3}>={threshold_p_value}. Cannot reject null-hypothesis. Fine-tuned model F1-score STDs cannot be proved to be statistically less than subject-specific model F1-score STDs. Fine-tuned model is not proved to be more stable."
    print(w_std_conclusion)

    print('The end.')