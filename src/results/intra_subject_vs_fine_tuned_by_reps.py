import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import json
from libemg.datasets import *
from typing import cast

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from libemg_3dc.utils.training_experiments import TrainingExperiments, NeuralNetworkSingleSubjectTrainingExperiment, NeuralNetworkFineTunedTrainigExperiment

df_rows = []

intra_subject_8_reps_experiments = TrainingExperiments.load(path='libemg_3dc/prove_pretraining_helps/single_subject/cnn_v1_results(8 reps).json')
intra_subject_8_reps_experiments = [cast(NeuralNetworkSingleSubjectTrainingExperiment, result) for result in intra_subject_8_reps_experiments.data if isinstance(result, NeuralNetworkSingleSubjectTrainingExperiment)]
intra_subject_8_reps_f1_scores = [result.test_result["f1_score"] for result in intra_subject_8_reps_experiments]
df_rows.extend([{"reps": "8", "approach": "Intra-subject", "f1_score": f1_score} for f1_score in intra_subject_8_reps_f1_scores])

intra_subject_6_reps_experiments = TrainingExperiments.load(path='libemg_3dc/prove_pretraining_helps/single_subject/cnn_v1_results(6 reps).json')
intra_subject_6_reps_experiments = [cast(NeuralNetworkSingleSubjectTrainingExperiment, result) for result in intra_subject_6_reps_experiments.data if isinstance(result, NeuralNetworkSingleSubjectTrainingExperiment)]
intra_subject_6_reps_f1_scores = [result.test_result["f1_score"] for result in intra_subject_6_reps_experiments]
df_rows.extend([{"reps": "6", "approach": "Intra-subject", "f1_score": f1_score} for f1_score in intra_subject_6_reps_f1_scores])

intra_subject_4_reps_experiments = TrainingExperiments.load(path='libemg_3dc/prove_pretraining_helps/single_subject/cnn_v1_results(4 reps).json')
intra_subject_4_reps_experiments = [cast(NeuralNetworkSingleSubjectTrainingExperiment, result) for result in intra_subject_4_reps_experiments.data if isinstance(result, NeuralNetworkSingleSubjectTrainingExperiment)]
intra_subject_4_reps_f1_scores = [result.test_result["f1_score"] for result in intra_subject_4_reps_experiments]
df_rows.extend([{"reps": "4", "approach": "Intra-subject", "f1_score": f1_score} for f1_score in intra_subject_4_reps_f1_scores])

intra_subject_3_reps_experiments = TrainingExperiments.load(path='libemg_3dc/prove_pretraining_helps/single_subject/cnn_v1_results(4 reps).json')
intra_subject_3_reps_experiments = [cast(NeuralNetworkSingleSubjectTrainingExperiment, result) for result in intra_subject_3_reps_experiments.data if isinstance(result, NeuralNetworkSingleSubjectTrainingExperiment)]
intra_subject_3_reps_f1_scores = [result.test_result["f1_score"] for result in intra_subject_3_reps_experiments]
df_rows.extend([{"reps": "3", "approach": "Intra-subject", "f1_score": f1_score} for f1_score in intra_subject_3_reps_f1_scores])

finetune_with_fc_reset_8_reps_experiments = TrainingExperiments.load(path='libemg_3dc/prove_pretraining_helps/fine_tuned/cnn_v1_results(finetune_with_fc_reset 8 reps).json')
finetune_with_fc_reset_8_reps_experiments = [cast(NeuralNetworkFineTunedTrainigExperiment, result) for result in finetune_with_fc_reset_8_reps_experiments.data if isinstance(result, NeuralNetworkFineTunedTrainigExperiment)]
finetune_with_fc_reset_8_reps_f1_scores = [experiment.test_result["f1_score"] for experiment in finetune_with_fc_reset_8_reps_experiments]
df_rows.extend([{"reps": "8", "approach": "TL with FC reset", "f1_score": f1_score} for f1_score in finetune_with_fc_reset_8_reps_f1_scores])

finetune_with_fc_reset_6_reps_experiments = TrainingExperiments.load(path='libemg_3dc/prove_pretraining_helps/fine_tuned/cnn_v1_results(finetune_with_fc_reset 6 reps).json')
finetune_with_fc_reset_6_reps_experiments = [cast(NeuralNetworkFineTunedTrainigExperiment, result) for result in finetune_with_fc_reset_6_reps_experiments.data if isinstance(result, NeuralNetworkFineTunedTrainigExperiment)]
finetune_with_fc_reset_6_reps_f1_scores = [experiment.test_result["f1_score"] for experiment in finetune_with_fc_reset_6_reps_experiments]
df_rows.extend([{"reps": "6", "approach": "TL with FC reset", "f1_score": f1_score} for f1_score in finetune_with_fc_reset_6_reps_f1_scores])

finetune_with_fc_reset_4_reps_experiments = TrainingExperiments.load(path='libemg_3dc/prove_pretraining_helps/fine_tuned/cnn_v1_results(finetune_with_fc_reset 4 reps).json')
finetune_with_fc_reset_4_reps_experiments = [cast(NeuralNetworkFineTunedTrainigExperiment, result) for result in finetune_with_fc_reset_4_reps_experiments.data if isinstance(result, NeuralNetworkFineTunedTrainigExperiment)]
finetune_with_fc_reset_4_reps_f1_scores = [experiment.test_result["f1_score"] for experiment in finetune_with_fc_reset_4_reps_experiments]
df_rows.extend([{"reps": "4", "approach": "TL with FC reset", "f1_score": f1_score} for f1_score in finetune_with_fc_reset_4_reps_f1_scores])

finetune_with_fc_reset_3_reps_experiments = TrainingExperiments.load(path='libemg_3dc/prove_pretraining_helps/fine_tuned/cnn_v1_results(finetune_with_fc_reset 3 reps).json')
finetune_with_fc_reset_3_reps_experiments = [cast(NeuralNetworkFineTunedTrainigExperiment, result) for result in finetune_with_fc_reset_3_reps_experiments.data if isinstance(result, NeuralNetworkFineTunedTrainigExperiment)]
finetune_with_fc_reset_3_reps_f1_scores = [experiment.test_result["f1_score"] for experiment in finetune_with_fc_reset_3_reps_experiments]
df_rows.extend([{"reps": "3", "approach": "TL with FC reset", "f1_score": f1_score} for f1_score in finetune_with_fc_reset_3_reps_f1_scores])


df = pd.DataFrame(columns=['reps', 'approach', 'f1_score'], data=df_rows)
print(df.info())
print(df.describe())
print(df.head(5))


plt.figure(figsize=(14, 8))
sns.set_theme(style="whitegrid")

# Main boxplot
ax = sns.boxplot(x='reps', y='f1_score', hue='approach', data=df,
                 palette='Greys', width=0.5, fliersize=4,
                 meanline=True, showmeans=True,
                 medianprops={'color': 'black', 'linewidth': 2, 'linestyle': '-'},
                 meanprops={'color': 'black', 'linewidth': 1, 'linestyle': '--'})

# Annotate mean and std above each box
ax.text(0-0.38, np.mean(intra_subject_8_reps_f1_scores), f'μ={np.mean(intra_subject_8_reps_f1_scores):.3f}\nσ={np.std(intra_subject_8_reps_f1_scores):.3f}', ha='center', va='center', fontsize=12, color='black')
ax.text(0+0.38, np.mean(finetune_with_fc_reset_8_reps_f1_scores), f'μ={np.mean(finetune_with_fc_reset_8_reps_f1_scores):.3f}\nσ={np.std(finetune_with_fc_reset_8_reps_f1_scores):.3f}', ha='center', va='center', fontsize=12, color='black')

ax.text(1-0.38, np.mean(intra_subject_6_reps_f1_scores), f'μ={np.mean(intra_subject_6_reps_f1_scores):.3f}\nσ={np.std(intra_subject_6_reps_f1_scores):.3f}', ha='center', va='center', fontsize=12, color='black')
ax.text(1+0.38, np.mean(finetune_with_fc_reset_6_reps_f1_scores), f'μ={np.mean(finetune_with_fc_reset_6_reps_f1_scores):.3f}\nσ={np.std(finetune_with_fc_reset_6_reps_f1_scores):.3f}', ha='center', va='center', fontsize=12, color='black')

ax.text(2-0.38, np.mean(intra_subject_4_reps_f1_scores), f'μ={np.mean(intra_subject_4_reps_f1_scores):.3f}\nσ={np.std(intra_subject_4_reps_f1_scores):.3f}', ha='center', va='center', fontsize=12, color='black')
ax.text(2+0.38, np.mean(finetune_with_fc_reset_4_reps_f1_scores), f'μ={np.mean(finetune_with_fc_reset_4_reps_f1_scores):.3f}\nσ={np.std(finetune_with_fc_reset_4_reps_f1_scores):.3f}', ha='center', va='center', fontsize=12, color='black')

ax.text(3-0.38, np.mean(intra_subject_3_reps_f1_scores), f'μ={np.mean(intra_subject_3_reps_f1_scores):.3f}\nσ={np.std(intra_subject_3_reps_f1_scores):.3f}', ha='center', va='center', fontsize=12, color='black')
ax.text(3+0.38, np.mean(finetune_with_fc_reset_3_reps_f1_scores), f'μ={np.mean(finetune_with_fc_reset_3_reps_f1_scores):.3f}\nσ={np.std(finetune_with_fc_reset_3_reps_f1_scores):.3f}', ha='center', va='center', fontsize=12, color='black')

# Avoid duplicate legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[0:2], labels[0:2], title='', loc='lower right')

# Axis formatting
# ax.set_title("Accuracy Distribution by Method and Training Cycle", fontsize=16)
ax.set_xlabel("Number of Training Repetitions", fontsize=16)
ax.set_ylabel("F1-score", fontsize=16)
plt.tight_layout()
plt.show()