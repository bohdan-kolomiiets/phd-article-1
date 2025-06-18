import os
import random
import sys
import time
from typing import Iterator, cast
import shutil
import numpy as np
import sklearn
import sklearn.metrics
from sklearn.model_selection import LeaveOneOut
import torch
from torch.utils.tensorboard import SummaryWriter
from libemg.datasets import *
from libemg.emg_predictor import EMGClassifier


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from utils.print_with_date import printd
from utils.collection_utils import group_by
from utils.model_checkpoint import ModelCheckpoint
from utils.libemg_deep_learning import make_data_loader
from utils.libemg_offline_data_handler_utils import get_standardization_params, apply_standardization_params, split_data_on_3_sets_by_reps
from utils.neural_networks.libemg_cnn_v1 import CNN_V1 as CNN
from utils.subject_repetitions_cross_validation import generate_3_repetitions_folds
from libemg_3dc.utils.training_experiments import TrainingExperiment, TrainingExperiments, NeuralNetworkOtherSubjectsTrainingExperiment, NeuralNetworkFineTunedTrainigExperiment

def add_model_graph_to_tensorboard(model, dataloader, tensorboard_writer):
    data, labels = next(iter(dataloader))
    data = CNN._try_move_to_accelerator(data)
    tensorboard_writer.add_graph(model, data)
    tensorboard_writer.flush()

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(mode=True, warn_only=False)

def create_tensorboard_writer(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    return SummaryWriter(folder_path)


def create_model_checkpoint(path, standardization_mean, standardization_std, n_output, n_channels, n_samples, n_filters):
    model_checkpoint = ModelCheckpoint(path=path, verbose=False)
    model_checkpoint.set_config("n_output", n_output)
    model_checkpoint.set_config("n_channels", n_channels)
    model_checkpoint.set_config("n_samples", n_samples)
    model_checkpoint.set_config("n_filters", n_filters)
    model_checkpoint.set_config("standardization_mean", standardization_mean)
    model_checkpoint.set_config("standardization_std", standardization_std)
    return model_checkpoint

def create_tensorboard_writer(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    return SummaryWriter(folder_path)

def create_log_callback(training_result: TrainingExperiment):
    
    tensorboard_writer = create_tensorboard_writer(folder_path=os.path.join("tensorboard", 'libemg_3dc', training_result.id))

    def log_callback(epoch, epoch_trloss, epoch_tracc, epoch_valoss, epoch_vaacc):

        printd(f"{epoch+1}: trloss:{epoch_trloss:.2f}  tracc:{epoch_tracc:.2f}  valoss:{epoch_valoss:.2f}  vaacc:{epoch_vaacc:.2f}")

        training_result.save_training_data({
            "epoch": epoch+1,
            "training_accuracy": epoch_tracc,
            "training_loss": epoch_trloss,
            "validation_loss": epoch_valoss,
            "validation_accuracy": epoch_vaacc
        })

        tensorboard_writer.add_scalars(main_tag='Training loss', tag_scalar_dict= { 'training_loss': epoch_trloss }, global_step=epoch+1)
        tensorboard_writer.add_scalars(main_tag='Training accuracy', tag_scalar_dict= { 'training_accuracy': epoch_tracc }, global_step=epoch+1)
        tensorboard_writer.add_scalars(main_tag='Validation loss', tag_scalar_dict= { 'validation_loss': epoch_valoss }, global_step=epoch+1)
        tensorboard_writer.add_scalars(main_tag='Validation accuracy', tag_scalar_dict= { 'validation_accuracy': epoch_vaacc }, global_step=epoch+1)
        tensorboard_writer.flush()
        
    return log_callback


def generate_cross_validation_folds(all_subject_ids: list[int], best_pre_training_experiments: dict[int, dict]) -> Iterator["NeuralNetworkFineTunedTrainigExperiment"]:
    for subject_id in all_subject_ids:
        # repetition_folds = generate_3_repetitions_folds(all_repetitions=[1,2,3,4,5,6,7,8])
        # repetition_folds = generate_3_repetitions_folds(all_repetitions=[1,2,3,4,5,6])
        # repetition_folds = generate_3_repetitions_folds(all_repetitions=[1,2,3,4])
        repetition_folds = generate_3_repetitions_folds(all_repetitions=[1,2,3])
        for repetition_fold in repetition_folds:
            training_result: NeuralNetworkFineTunedTrainigExperiment = NeuralNetworkFineTunedTrainigExperiment.create(
                subject_id=subject_id,
                pre_train_experiment_id=best_pre_training_experiments[subject_id].id,
                train_reps=repetition_fold['train_reps'], 
                validate_reps=repetition_fold['validate_reps'], 
                test_reps=repetition_fold['test_reps'])
            yield training_result


def find_best_experiments_per_test_subject(other_subjects_experiments) -> dict[int, dict]:
    experiments_per_test_subject = group_by(other_subjects_experiments, key_selector= lambda experiment: experiment.test_subject_ids[0])
    best_pre_training_experiments: dict[int, dict] = {}
    for test_subject_id, test_subject_experiments in experiments_per_test_subject.items():
        best_experiment = max(test_subject_experiments, key=lambda experiment: experiment.training_subjects_test_result["f1_score"])
        best_pre_training_experiments[test_subject_id] = best_experiment
    return best_pre_training_experiments

seed = 123

num_subjects = 22
num_epochs = 50
batch_size = 64

# Adam optimizer params
adam_learning_rate = 1e-3
adam_weight_decay=0 # 1e-5

transfer_strategy = 'finetune_with_fc_reset'

other_subjects_experiments = TrainingExperiments.load(path='libemg_3dc/prove_pretraining_helps/other_subjects/cnn_v1_results(ready).json')

fine_tuned_experiments = TrainingExperiments.load(path='libemg_3dc/prove_pretraining_helps/fine_tuned/cnn_v1_results.json')

# training_results.cleanup(
#     model_type=NeuralNetworkOtherSubjectsTrainingResult.model_type, 
#     experiment_type=NeuralNetworkOtherSubjectsTrainingResult.experiment_type)
           

if __name__ == "__main__":
    
    set_seed(seed)

    all_subject_ids = list(range(0,num_subjects))

    best_pre_training_experiments = find_best_experiments_per_test_subject(other_subjects_experiments.data)
    all_possible_fine_tune_experiments  = generate_cross_validation_folds(all_subject_ids, best_pre_training_experiments)

    dataset = get_dataset_list()['3DC']()
    odh_full = dataset.prepare_data(subjects=all_subject_ids)
    odh = odh_full['All']

    for fine_tuned_experiment in all_possible_fine_tune_experiments:
        
        if(fine_tuned_experiment in fine_tuned_experiments.data):
            continue

        start = time.perf_counter()

        pre_train_experiment = best_pre_training_experiments[fine_tuned_experiment.subject_id]
        pre_trained_model_state, pre_trained_model_config = ModelCheckpoint.load(f'libemg_3dc/checkpoints/{pre_train_experiment.id}.pt')

        subject_measurements  = odh.isolate_data("subjects", [fine_tuned_experiment.subject_id])

        (train_measurements, validate_measurements, test_measurements) = split_data_on_3_sets_by_reps(
            odh=subject_measurements, 
            train_reps=fine_tuned_experiment.train_reps, 
            validate_reps=fine_tuned_experiment.validate_reps, 
            test_reps=fine_tuned_experiment.test_reps)

        # apply standardization
        standardization_mean = pre_trained_model_config["standardization_mean"]
        standardization_std = pre_trained_model_config["standardization_std"]
        train_measurements = apply_standardization_params(train_measurements, standardization_mean, standardization_std)
        validate_measurements = apply_standardization_params(validate_measurements, standardization_mean, standardization_std)
        
        # perform windowing
        train_windows, train_metadata = train_measurements.parse_windows(200,100)
        validate_windows, validate_metadata = validate_measurements.parse_windows(200,100)
                    
        # fine-tune pre-trained model
        generator = torch.Generator().manual_seed(seed)
        pre_trained_model = CNN(
            n_output=pre_trained_model_config["n_output"], 
            n_channels=pre_trained_model_config["n_channels"], 
            n_samples=pre_trained_model_config["n_samples"], 
            n_filters=pre_trained_model_config["n_filters"], 
            generator=generator)
        pre_trained_model.load_state_dict(pre_trained_model_state)
        pre_trained_model.apply_transfer_strategy(strategy=transfer_strategy)

        model_checkpoint = create_model_checkpoint(
            path=f"libemg_3dc/checkpoints/{fine_tuned_experiment.id}.pt", 
            standardization_mean=pre_trained_model_config["standardization_mean"], standardization_std=pre_trained_model_config["standardization_std"], 
            n_output=pre_trained_model_config["n_output"], n_channels=pre_trained_model_config["n_channels"], n_samples=pre_trained_model_config["n_samples"], n_filters=pre_trained_model_config["n_filters"])
        dataloader_dictionary = {
            "training_dataloader": make_data_loader(train_windows, train_metadata["classes"], batch_size=batch_size, generator=generator),
            "validation_dataloader": make_data_loader(validate_windows, validate_metadata["classes"], batch_size=batch_size, generator=generator)
            }
        log_callback = create_log_callback(fine_tuned_experiment)
        pre_trained_model.fit(dataloader_dictionary, num_epochs, adam_learning_rate, adam_weight_decay, verbose=True, 
                model_checkpoint=model_checkpoint, training_log_callback=log_callback)

        # load fine-tuned model
        generator = torch.Generator().manual_seed(seed)
        fine_tuned_model_state, fine_tuned_model_config = model_checkpoint.load_best_model_config()
        fine_tuned_model = CNN(
            n_output=fine_tuned_model_config["n_output"], 
            n_channels=fine_tuned_model_config["n_channels"], 
            n_samples=fine_tuned_model_config["n_samples"], 
            n_filters=fine_tuned_model_config["n_filters"], 
            generator=generator)
        fine_tuned_model.load_state_dict(fine_tuned_model_state)
        fine_tuned_classifier = EMGClassifier(None)
        fine_tuned_classifier.model = fine_tuned_model

        # test
        test_measurements = apply_standardization_params(test_measurements, mean_by_channels=fine_tuned_model_config["standardization_mean"], std_by_channels=fine_tuned_model_config["standardization_std"])
        test_windows, test_metadata = test_measurements.parse_windows(200,100)
        predicted_classes, class_probabilities = fine_tuned_classifier.run(test_windows)
        print('Metrics: \n', sklearn.metrics.classification_report(y_true=test_metadata["classes"], y_pred=predicted_classes, output_dict=False))
        
        # save results
        fine_tuned_experiment.save_test_result(
            classification_report=sklearn.metrics.classification_report(y_true=test_metadata["classes"], y_pred=predicted_classes, output_dict=True))
        
        end = time.perf_counter()
        fine_tuned_experiment.set_processing_duration(duration=int(end-start))

        fine_tuned_experiments.append(fine_tuned_experiment)