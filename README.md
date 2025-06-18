# Codebase for article "Improving EMG Signal Classification with Transfer Learning Under Low-Data and Cross-Subject Conditions"

This repository contains code used to fetch data. from sEMG dataset, build CNN, execute experiments, and process results.

### CNN model implementation

Code of Convolutional Neural Network (CNN) used in this study is located in file `src/utils/libemg/neural_networks/libemg_cnn_v1.py`.

### Evaluated training approaches

Under `src/experiments` folder, you will find code for each of three evaluated training approaches: "Intra-subject training", "Inter-subject training", "Transfer learning".

Source code for training approach refered to as **"Intra-subject training"** is located at folder `src/experiments/single_subject`. The script `cnn_v1.py` in this folder contains code to fetch data, train and test CNN model on same subject one by one, and store model metrics. Files called `cnn_v1_results(X reps).json` contain model metrics for cross-validation folds when using 8,6,4,3 repetitions for training respecitvely.

Source code for training approach refered to as **"Inter-subject training"** is located at folder `src/experiments/other_subjects`. The script `cnn_v1.py` in this folder contains code to fetch data, train CNN model on all subjects expect one and then test on the subject excluded, and store model metrics. File called `cnn_v1_results(ready).json` contains model metrics for cross-validation folds when using 8 repetitions for training respecitvely.

Source code for training approach refered to as **"Transfer learning"** is located at folder `src/experiments/fine_tuned`. The script `cnn_v1.py` in this folder contains code to fetch data, get the best pre-trained CNN model excluding target subject, fine-tune this model on the part of data from the target subject, and then test model perfomance on data of this subject that is left. The Script `cnn_v1.py` was used for both fine-tune strategies "with FC reset" and "without FC reset" which was controlled by variable `transfer_strategy`. File called `cnn_v1_results(finetune_without_fc_reset 8 reps).json` contains training metrics for cross validation folds when fine tune strategy "without FC reset" was used wth all 8 repetitions. Files called `cnn_v1_results(finetune_with_fc_reset X reps).json` contain training metrics for cross validation folds when fine tune strategy "with FC reset" was used wth 8,6,4,3 repetitions respectively. 

### Results processing

#### Model metrics statistics

Folder `results` contains scripts used to process obtained model metrics for each of training approach.
The script `accuracy_by_approach_8_reps.py` calculates data for box and whisker plot of accuracies for each training approach and draws this plot. 
The script `intra_subject_vs_fine_tuned_by_reps.py` calculates data for box and whisker plot to comapre accuracies between "Intra-subject training" and "Transfer learning with FC reset" using a different number of repetitions (8,6,4,3).

#### Hypothesis testing

The `hypothesis_testing` folder contains scripts to prepare data and run T-test and Wilcoxon signed-rank tests on this data.