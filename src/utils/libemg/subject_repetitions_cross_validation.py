import random
import itertools
from sklearn.model_selection import KFold, LeaveOneOut
import numpy as np
from typing import Generator

def generate_random_folds(items, n_splits, n_items):
    for _ in range(n_splits):
        fold_indexes = random.sample(range(len(items)), n_items)
        yield fold_indexes

def select_test_repetition_folds(reps: np.ndarray[int]) -> Generator[tuple[np.ndarray[int], np.ndarray[int]], None, None]:
    # for test_indexes in generate_random_folds(reps, n_splits=3, n_items=1):
    #     train_indexes = np.setdiff1d(range(len(reps)), test_indexes)
    #     yield reps[train_indexes], reps[test_indexes]
        
    loo = LeaveOneOut()
    for train_indexes, test_indexes in loo.split(reps):
        yield reps[train_indexes], reps[test_indexes]


def select_validate_repetition_folds(reps: np.ndarray[int]) -> Generator[tuple[np.ndarray[int], np.ndarray[int]], None, None]:
    # for validate_indexes in generate_random_folds(reps, n_splits=3, n_items=1):
    #     train_indexes = np.setdiff1d(range(len(reps)), validate_indexes)
    #     yield reps[train_indexes], reps[validate_indexes]
        
    loo = LeaveOneOut()
    for train_indexes, validate_indexes in loo.split(reps):
        yield reps[train_indexes], reps[validate_indexes]


def generate_3_repetitions_folds(all_repetitions: list[int], test_repetitions: list[int] = None) -> list[dict]:
    """
    all_reps = np.array([1,2,3,4,5,6,7,8])
    LOO: LOO for test reps(8), LOO for validate reps (7) = 56
    Custom: 3 folds of 2 test reps, 3 folds of 2 validate reps = 9
    Custom: 3 folds of 1 test rep, 3 folds of 1 validate reps = 9
    Custom + LDO: 3 folds of 1 test rep, LOO for validate reps (7) = 21 !!!
    """
    folds = []
    
    for (non_test_reps, test_reps) in select_test_repetition_folds(np.array(all_repetitions)) if test_repetitions is None else [(np.array([rep for rep in all_repetitions if rep not in test_repetitions]), np.array(test_repetitions))]:
        for (train_reps, validate_reps) in select_validate_repetition_folds(non_test_reps):
            folds.append({
                'train_reps': [int(rep) for rep in train_reps],
                'validate_reps': [int(rep) for rep in validate_reps],
                'test_reps': [int(rep) for rep in test_reps] 
            })
    return folds

def generate_2_repetitions_folds(all_repetitions: list[int], validation_reps_count = 1):
    """
    all_reps = np.array([1,2,3,4,5,6,7,8])
    LOO: LOO for test reps(8), LOO for validate reps (7) = 56
    Custom: 3 folds of 2 test reps, 3 folds of 2 validate reps = 9
    Custom: 3 folds of 1 test rep, 3 folds of 1 validate reps = 9
    Custom + LDO: 3 folds of 1 test rep, LOO for validate reps (7) = 21 !!!
    """
    validation_combinations = list(itertools.combinations(all_repetitions, validation_reps_count))

    folds = []

    for validation_combination in validation_combinations:
        train_reps = [r for r in all_repetitions if r not in validation_combination]
        folds.append({
            'train_reps': train_reps,
            'validate_reps': list(validation_combination)
        })
    return folds