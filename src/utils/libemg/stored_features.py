import os
import numpy as np
import pandas as pd


def read_features(directory_path):
    
    file_names = os.listdir(directory_path)

    feature_file_names = [file_name for file_name in file_names if "metadata" not in file_name]
    features = {}
    for feature_file_name in feature_file_names:
        feature_name = feature_file_name.removesuffix(".csv")
        features[feature_name] = pd.read_csv(f"{directory_path}/{feature_file_name}").values
    
    metadata_file_names = [file_name for file_name in file_names if "metadata" in file_name]
    metadata = {}
    for metadata_file_name in metadata_file_names:
        metadata_name = metadata_file_name.removeprefix("metadata_").removesuffix(".csv")
        metadata[metadata_name] = pd.read_csv(f"{directory_path}/{metadata_file_name}").values
        
    return metadata, features

def concatenate_feature_sets(set1: dict[str, np.ndarray], set2: dict[str, np.ndarray]):
    feature_names = set1.keys()
    concatenated = {}
    for feature_name in feature_names:
        concatenated[feature_name] = np.concatenate((set1[feature_name], set2[feature_name]), axis=0)
    return concatenated
