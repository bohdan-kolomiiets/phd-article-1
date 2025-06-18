import os
import random
import sys
import time
import shutil
from typing import Iterator
import numpy as np
import sklearn
import sklearn.metrics
import torch
from torch.utils.tensorboard import SummaryWriter
from libemg.datasets import *
from libemg.emg_predictor import EMGClassifier


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.utils.common.print_with_date import printd
from src.utils.common.collection_utils import group_by
from src.utils.common.model_checkpoint import ModelCheckpoint
from src.utils.libemg.libemg_deep_learning import make_data_loader
from src.utils.libemg.libemg_offline_data_handler_utils import get_standardization_params, apply_standardization_params, split_data_on_3_sets_by_reps
from src.utils.libemg.neural_networks.libemg_cnn_v1 import CNN_V1 as CNN
from src.utils.libemg.subject_repetitions_cross_validation import generate_3_repetitions_folds
from src.utils.libemg.training_experiments import TrainingExperiment, TrainingExperiments, NeuralNetworkSingleSubjectTrainingExperiment

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


def add_model_graph_to_tensorboard(tensorboard_writer, model, dataloader):
    data, labels = next(iter(dataloader))
    data = CNN._try_move_to_accelerator(data)
    tensorboard_writer.add_graph(model, data)
    tensorboard_writer.flush()

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

def generate_cross_validation_folds(all_subject_ids: list[int]) -> Iterator["NeuralNetworkSingleSubjectTrainingExperiment"]:
    for subject_id in all_subject_ids:
        # repetition_folds = generate_3_repetitions_folds(all_repetitions=[1,2,3,4,5,6,7,8])
        # repetition_folds = generate_3_repetitions_folds(all_repetitions=[1,2,3,4,5,6])
        # repetition_folds = generate_3_repetitions_folds(all_repetitions=[1,2,3,4])
        repetition_folds = generate_3_repetitions_folds(all_repetitions=[1,2,3])
        for repetition_fold in repetition_folds:
            experiment = NeuralNetworkSingleSubjectTrainingExperiment.create(
                subject_id=subject_id, 
                train_reps=repetition_fold['train_reps'], 
                validate_reps=repetition_fold['validate_reps'], 
                test_reps=repetition_fold['test_reps'])
            yield experiment


seed = 123

num_subjects = 2
num_epochs = 50
batch_size = 64

# Adam optimizer params
adam_learning_rate = 1e-3
adam_weight_decay=0 # 1e-5

processed_experiments = TrainingExperiments.load(path='libemg_3dc/prove_pretraining_helps/single_subject/cnn_v1_results.json')

# training_results.cleanup(
#     model_type=NeuralNetworkSingleSubjectTrainingResult.model_type, 
#     experiment_type=NeuralNetworkSingleSubjectTrainingResult.experiment_type)

if __name__ == "__main__":
    
    set_seed(seed)

    all_subject_ids = list(range(0,num_subjects))

    dataset = get_dataset_list()['3DC']()
    odh_full = dataset.prepare_data(subjects=all_subject_ids)
    odh = odh_full['All']

    all_possible_experiments  = generate_cross_validation_folds(all_subject_ids)

    for experiment in all_possible_experiments:
        
        if(experiment in processed_experiments.data):
            continue

        start = time.perf_counter()

        subject_measurements  = odh.isolate_data("subjects", [experiment.subject_id])

        (train_measurements, validate_measurements, test_measurements) = split_data_on_3_sets_by_reps(
            odh=subject_measurements, 
            train_reps=experiment.train_reps, 
            validate_reps=experiment.validate_reps, 
            test_reps=experiment.test_reps)

        # apply standardization
        standardization_mean, standardization_std = get_standardization_params(train_measurements)
        train_measurements = apply_standardization_params(train_measurements, standardization_mean, standardization_std)
        validate_measurements = apply_standardization_params(validate_measurements, standardization_mean, standardization_std)
        
        # perform windowing
        train_windows, train_metadata = train_measurements.parse_windows(200,100)
        validate_windows, validate_metadata = validate_measurements.parse_windows(200,100)
        
        # train
        n_output = len(np.unique(np.vstack(train_metadata['classes'])))
        n_channels = train_windows.shape[1]
        n_samples = train_windows.shape[2]
        n_filters = batch_size
        log_callback = create_log_callback(experiment)
        generator = torch.Generator().manual_seed(seed)
        model = CNN(n_output, n_channels, n_samples, n_filters = batch_size, generator=generator)
        model_checkpoint = create_model_checkpoint(
            f'libemg_3dc/checkpoints/{experiment.id}.pt', standardization_mean, standardization_std, n_output, n_channels, n_samples, n_filters)
        dataloader_dictionary = {
            "training_dataloader": make_data_loader(train_windows, train_metadata["classes"], batch_size=batch_size, generator=generator),
            "validation_dataloader": make_data_loader(validate_windows, validate_metadata["classes"], batch_size=batch_size, generator=generator)
            }
        model.fit(dataloader_dictionary, num_epochs, adam_learning_rate, adam_weight_decay, verbose=True, 
                model_checkpoint=model_checkpoint, training_log_callback=log_callback)

        # load trained model            
        model_state, model_config = model_checkpoint.load_best_model_config()
        model.load_state_dict(model_state)
        classifier = EMGClassifier(None)
        classifier.model = model

        tensorboard_writer = create_tensorboard_writer(folder_path=os.path.join("tensorboard2", 'libemg_3dc', 'graph'))
        add_model_graph_to_tensorboard(tensorboard_writer, model, dataloader_dictionary["training_dataloader"])

        # test        
        test_measurements = apply_standardization_params(test_measurements, mean_by_channels=model_config["standardization_mean"], std_by_channels=model_config["standardization_std"])
        test_windows, test_metadata = test_measurements.parse_windows(200,100)
        predicted_classes, class_probabilities = classifier.run(test_windows)
        print('Training finished.\nMetrics: \n', sklearn.metrics.classification_report(y_true=test_metadata['classes'], y_pred=predicted_classes, output_dict=False))
        
        # save results
        experiment.save_test_result(classification_report=sklearn.metrics.classification_report(y_true=test_metadata['classes'], y_pred=predicted_classes, output_dict=True))
        
        end = time.perf_counter()
        experiment.set_processing_duration(duration=int(end-start))

        processed_experiments.append(experiment)