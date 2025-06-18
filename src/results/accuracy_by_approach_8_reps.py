import os
import sys
import numpy as np
import json
from libemg.datasets import *
from typing import cast
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from libemg_3dc.utils.training_experiments import TrainingExperiments, NeuralNetworkSingleSubjectTrainingExperiment, NeuralNetworkOtherSubjectsTrainingExperiment, NeuralNetworkFineTunedTrainigExperiment

inter_subjects_experiments = TrainingExperiments.load(path='libemg_3dc/prove_pretraining_helps/other_subjects/cnn_v1_results(ready).json')
inter_subjects_experiments = [cast(NeuralNetworkOtherSubjectsTrainingExperiment, result) for result in inter_subjects_experiments.data if isinstance(result, NeuralNetworkOtherSubjectsTrainingExperiment)]
inter_subjects_f1_scores = [experiment.test_subjects_test_result["f1_score"] for experiment in inter_subjects_experiments]
inter_subjects_f1_scores_mean = np.mean(inter_subjects_f1_scores) # 0.382
inter_subjects_f1_scores_std = np.std(inter_subjects_f1_scores) # 0.149

intra_subject_8_reps_experiments = TrainingExperiments.load(path='libemg_3dc/prove_pretraining_helps/single_subject/cnn_v1_results(8 reps).json')
intra_subject_8_reps_experiments = [cast(NeuralNetworkSingleSubjectTrainingExperiment, result) for result in intra_subject_8_reps_experiments.data if isinstance(result, NeuralNetworkSingleSubjectTrainingExperiment)]
intra_subject_8_reps_f1_scores = [result.test_result["f1_score"] for result in intra_subject_8_reps_experiments]
intra_subject_8_reps_f1_scores_mean = np.mean(intra_subject_8_reps_f1_scores) # 0.869
intra_subject_8_reps_f1_scores_std = np.std(intra_subject_8_reps_f1_scores) # 0.089

finetune_with_fc_reset_8_reps_experiments = TrainingExperiments.load(path='libemg_3dc/prove_pretraining_helps/fine_tuned/cnn_v1_results(finetune_with_fc_reset 8 reps).json')
finetune_with_fc_reset_8_reps_experiments = [cast(NeuralNetworkFineTunedTrainigExperiment, result) for result in finetune_with_fc_reset_8_reps_experiments.data if isinstance(result, NeuralNetworkFineTunedTrainigExperiment)]
finetune_with_fc_reset_8_reps_f1_scores = [experiment.test_result["f1_score"] for experiment in finetune_with_fc_reset_8_reps_experiments]
finetune_with_fc_reset_8_reps_f1_scores_mean = np.mean(finetune_with_fc_reset_8_reps_f1_scores) # 0.907
finetune_with_fc_reset_8_reps_f1_scores_std = np.std(finetune_with_fc_reset_8_reps_f1_scores) # 0.074

finetune_without_fc_reset_8_reps_experiments = TrainingExperiments.load(path='libemg_3dc/prove_pretraining_helps/fine_tuned/cnn_v1_results(finetune_without_fc_reset 8 reps).json')
finetune_without_fc_reset_8_reps_experiments = [cast(NeuralNetworkFineTunedTrainigExperiment, result) for result in finetune_without_fc_reset_8_reps_experiments.data if isinstance(result, NeuralNetworkFineTunedTrainigExperiment)]
finetune_without_fc_reset_8_reps_f1_scores = [experiment.test_result["f1_score"] for experiment in finetune_without_fc_reset_8_reps_experiments]
finetune_without_fc_reset_8_reps_f1_scores_mean = np.mean(finetune_without_fc_reset_8_reps_f1_scores) # 0.907
finetune_without_fc_reset_8_reps_f1_scores_std = np.std(finetune_without_fc_reset_8_reps_f1_scores) # 0.074

f1_scores = [
    inter_subjects_f1_scores,
    intra_subject_8_reps_f1_scores,
    finetune_without_fc_reset_8_reps_f1_scores,
    finetune_with_fc_reset_8_reps_f1_scores
]
labels = ['Inter-subject', 'Intra-subect', 'FT without FC reset', 'FT with FC reset']

fig, ax = plt.subplots(figsize=(14, 8))
bplot = ax.boxplot(f1_scores,
                   patch_artist=True,  # fill with color
                   tick_labels=labels,
                   showfliers=True,  # show outliers
                   vert=True,  # vertical box alignment
                   widths=0.5,  # width of the boxes
                   meanline=True,
                   showmeans=True,  # show mean line
                   medianprops={'color': 'black', 'linewidth': 2, 'linestyle': '-'},  # style of the median line
                   meanprops={'color': 'black', 'linewidth': 1, 'linestyle': '--'},  # style of the mean line
                   showbox=True)  # will be used to label x-ticks

for box in bplot['boxes']:
    box.set_facecolor('lightgray')

# Annotate mean and std above each box
for i, group in enumerate(f1_scores, start=1):
    mean = np.mean(group)
    std = np.std(group)
    ax.text(i-0.37, mean, f'μ={mean:.3f}\nσ={std:.3f}', ha='center', va='bottom', fontsize=12, color='black')
    
# Add y-axis grid with 0.1 step
ax.set_yticks(np.arange(0.0, 1.05, 0.1))
ax.set_ylim(0.0, 1.05)
ax.grid(True, axis='y', linestyle='--', alpha=0.5)

# Final touches
ax.set_xticklabels(labels, fontsize=16)
ax.set_xlabel("Training approaches", fontsize=16)
ax.set_ylabel('F1-score', fontsize=16)
# ax.set_title('Accuracy of each training approach', fontsize=18)
plt.tight_layout()

plt.show()