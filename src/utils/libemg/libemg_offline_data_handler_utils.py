import numpy as np
from functools import reduce
from libemg.data_handler import OfflineDataHandler 

def get_standardization_params(odh: OfflineDataHandler):
    ''' Computes the mean and standard deviation of data contained in an OfflineDataHandler.

    Parameters
    ----------
    odh: OfflineDataHandler   
        The data that parameters will be computed from.

    Returns
    ------- 
    mean: np.ndarray
        channel-wise means.
    std:  np.ndarray
        channel-wise standard deviations.
    '''
    data = np.concatenate(odh.data)
    filter_mean = np.mean(data,axis=0)
    filter_std  = np.std(data, axis=0)
    assert (filter_std != 0).any()
    return filter_mean, filter_std


def apply_standardization_params(odh: OfflineDataHandler, mean_by_channels, std_by_channels):
    for record_index in range(len(odh.data)):
        odh.data[record_index] = (odh.data[record_index] - mean_by_channels) / std_by_channels
    return odh


def split_data_on_3_sets_by_reps(odh: OfflineDataHandler, train_reps: np.ndarray = None, validate_reps: np.ndarray = None, test_reps: np.ndarray = None):
    """
    all_repetition_ids = np.unique(np.concatenate(odh.reps)) # [0, 1, 2, 3]

    train_sets/validate_sets/test_sets expect array values from 1 to 8 

    returns (train_measurements, validate_measurements, test_measurements)
    """

    if train_reps is None or validate_reps is None or test_reps is None:
        train_measurements = odh.isolate_data("sets",[0])
        non_train_measurements  = odh.isolate_data("sets",[1])
        validate_measurements = non_train_measurements.isolate_data("reps",[0, 1])
        test_measurements = non_train_measurements.isolate_data("reps",[2, 3])
        return (train_measurements, validate_measurements, test_measurements)
    else:
        set1 = odh.isolate_data("sets",[0])
        set2  = odh.isolate_data("sets",[1])
        map = {
            1: set1.isolate_data("reps",[0]),
            2: set1.isolate_data("reps",[1]),
            3: set1.isolate_data("reps",[2]),
            4: set1.isolate_data("reps",[3]),
            5: set2.isolate_data("reps",[0]),
            6: set2.isolate_data("reps",[1]),
            7: set2.isolate_data("reps",[2]),
            8: set2.isolate_data("reps",[3])
        }
        train_measurements = reduce(lambda a, b: a + b, (map[set] for set in train_reps))
        validate_measurements = reduce(lambda a, b: a + b, (map[set] for set in validate_reps))
        test_measurements = reduce(lambda a, b: a + b, (map[set] for set in test_reps))
        return (train_measurements, validate_measurements, test_measurements)

def split_data_on_2_sets_by_reps(odh: OfflineDataHandler, train_reps: np.ndarray, validate_reps: np.ndarray):
    """
    all_repetition_ids = np.unique(np.concatenate(odh.reps)) # [0, 1, 2, 3]

    train_sets/test_sets expect array values from 1 to 8 

    returns (train_measurements, test_measurements)
    """
    set1 = odh.isolate_data("sets",[0])
    set2  = odh.isolate_data("sets",[1])
    map = {
        1: set1.isolate_data("reps",[0]),
        2: set1.isolate_data("reps",[1]),
        3: set1.isolate_data("reps",[2]),
        4: set1.isolate_data("reps",[3]),
        5: set2.isolate_data("reps",[0]),
        6: set2.isolate_data("reps",[1]),
        7: set2.isolate_data("reps",[2]),
        8: set2.isolate_data("reps",[3])
    }
    train_measurements = reduce(lambda a, b: a + b, (map[set] for set in train_reps))
    validate_measurements = reduce(lambda a, b: a + b, (map[set] for set in validate_reps))
    return (train_measurements, validate_measurements)
