import os
from typing import Union, Optional
from pathlib import Path
import uuid
import numpy as np
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from utils.dict_utils import get_nested

@dataclass(eq=False)
class TrainingExperiment(ABC):
    id: str
    model_type: str
    experiment_type: str
    processing_duration: Optional[float] = field(default=None, init=False)

    _registry = []

    @classmethod
    def register(cls, subclass):
        cls._registry.append(subclass)
        return subclass
    

    def __eq__(self, value):
        return isinstance(self, TrainingExperiment) and self.id == value.id
    
    def __hash__(self):
        return hash(self.id)
    
    def set_processing_duration(self, duration: float):
        self.processing_duration = duration

    @classmethod
    @abstractmethod
    def _can_create_from_json_dict(cls, json_dict: dict) -> bool:
        pass

    @classmethod
    @abstractmethod
    def _from_json_dict(cls, json_dict: dict) -> "TrainingExperiment":
        pass

    @abstractmethod
    def to_json_dict(self) -> dict:
        pass

    def __repr__(self):
        return self.id

    @classmethod
    def from_json_dict(cls, json_dict: dict) -> "TrainingExperiment":
        for subclass in cls._registry:
            try:
                if subclass._can_create_from_json_dict(json_dict):
                    return subclass._from_json_dict(json_dict)
            except Exception as e:
                print(f"Failed to create subclass: {e}")
                raise e 
        raise ValueError("No subclass can handle the provided JSON")
    



class TrainingExperiments:
    
    def __init__(self, path: Union[str, Path], results: list[TrainingExperiment]  = None):
        self.path = path
        self.data = results or []

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TrainingExperiments":

        results = []

        path = Path(path)
        if os.path.exists(path) and os.stat(path).st_size > 0: 
            with open(path, mode='r', encoding='utf-8') as file:
                json_dicts = json.load(fp=file) 

            for json_dict in json_dicts:
                try:
                    result = TrainingExperiment.from_json_dict(json_dict)
                    results.append(result)
                except Exception as e:
                    print(f'Skipped invalid result: {e}')
        
        return cls(path, results)


    def cleanup(self, model_type: str, experiment_type: str, indent: int = 2):
        self.data = [result for result in self.data if result.model_type != model_type and result.experiment_type != experiment_type]
        with open(self.path, "w") as file:
            json.dump([result.to_json_dict() for result in self.data], file, indent=2)
        

    def append(self, result: TrainingExperiment, indent: int = 2):
        
        self.data.append(result)

        # Convert the new object to a pretty-formatted JSON string
        obj_str = json.dumps(result.to_json_dict(), indent=indent)
        obj_str = "\n" + obj_str + "\n"  # add surrounding newlines for readability

        # Case 1: file does not exist or is empty → create new array with single object
        if not os.path.exists(self.path) or os.stat(self.path).st_size == 0:
            with open(self.path, "w", encoding="utf-8") as f:
                f.write("[")
                f.write(obj_str)
                f.write("]")
            return

        # Case 2: file exists and contains valid array → append inside the array
        with open(self.path, "rb+") as f:
            f.seek(0, os.SEEK_END)  # Go to the end of the file
            f.seek(-1, os.SEEK_CUR)  # Step back one character from the end
            last_char = f.read(1)    # Read the final character

            # Step backward until we find the true last non-whitespace character
            while last_char in b" \r\n\t":
                f.seek(-2, os.SEEK_CUR)
                last_char = f.read(1)

            # After this, the file pointer is right before the closing bracket `]`
            f.seek(-1, os.SEEK_CUR)
            pos = f.tell()  # Save this position to overwrite `]`
            f.seek(pos)

            # Peek at the character before `]` to check if the array is empty or not
            f.seek(-1, os.SEEK_CUR)
            char = f.read(1)
            f.seek(pos)  # Return to position before final bracket

            needs_comma = char != b"["  # If previous char is not `[`, we need a comma

            # Write the new object and restore the closing `]`
            if needs_comma:
                f.write(b",")  # Separate from previous object
            f.write(obj_str.encode("utf-8"))  # Write the new object
            f.write(b"]")  # Restore the closing bracket to complete the array



@TrainingExperiment.register
@dataclass(eq=False)
class UnknownTrainingExperiment(TrainingExperiment):
    model_type: str = field(init=False, default='unknown')
    experiment_type: str = field(init=False, default='unknown')

    def to_json_dict(self):
        return {
            "id": self.id
        }

    @classmethod
    def _can_create_from_json_dict(cls, json_dict: dict):
        id = json_dict.get("id", "")
        return f"experiment_type:unknown" in id 

    @classmethod
    def _from_json_dict(cls, json_dict: dict):
        metadata = dict(pair.split(":") for pair in json_dict["id"].split("."))
        return cls(
            id=json_dict["id"]
        )

@TrainingExperiment.register
@dataclass(eq=False)
class NeuralNetworkSingleSubjectTrainingExperiment(TrainingExperiment):
    subject_id: int
    train_reps: list[int]
    validate_reps: list[int]
    test_reps: list[int]
    training_data: list[str] = field(default_factory=list)
    test_result: dict = field(default=None)

    model_type: str = field(init=False, default='NN')
    experiment_type: str = field(init=False, default='single-subject')

    @classmethod
    def create(cls, 
               subject_id: int, 
               train_reps: list[int], validate_reps: list[int], test_reps: list[int]) -> "NeuralNetworkSingleSubjectTrainingExperiment":
        
        id_data = {
            "model_type": cls.model_type,
            "experiment_type": cls.experiment_type,
            "subject_id": subject_id,
            "train_reps": json.dumps(train_reps),
            "validate_reps": json.dumps(validate_reps),
            "test_reps": json.dumps(test_reps)
        }
        id = ".".join(f"{k}:{v}" for k, v in id_data.items())

        return cls(
            id=id,
            subject_id=subject_id, 
            train_reps=train_reps, 
            validate_reps=validate_reps, 
            test_reps=test_reps
        )

    def save_training_data(self, epoch_training_data: dict): 
        self.training_data.append(json.dumps(epoch_training_data))

    def save_test_result(self, classification_report: dict): 
        self.test_result = {
            "f1_score": classification_report['macro avg']['f1-score'],
            "report": json.dumps(classification_report)
        }

    def to_json_dict(self):
        return {
            "id": self.id,
            "processing_duration": self.processing_duration, 
            "training_data": self.training_data,
            "test_result": self.test_result
        }
    
    @classmethod
    def _can_create_from_json_dict(cls, json_dict: dict):
        id = json_dict.get("id", "")
        return f"model_type:{cls.model_type}" in id and f"experiment_type:{cls.experiment_type}" in id 

    @classmethod
    def _from_json_dict(cls, json_dict: dict):
        
        id_data = dict(pair.split(":") for pair in json_dict["id"].split("."))

        instance = cls(
            id=json_dict["id"],
            subject_id=int(id_data["subject_id"]), 
            train_reps=list(json.loads(id_data["train_reps"])), 
            validate_reps=list(json.loads(id_data["validate_reps"])), 
            test_reps=list(json.loads(id_data["test_reps"])), 
            training_data=json_dict.get("training_data", []),
            test_result=json_dict.get("test_result", None)
        )
        instance.set_processing_duration(json_dict.get("processing_duration", None))
        return instance
    

@TrainingExperiment.register
@dataclass(eq=False)
class NeuralNetworkOtherSubjectsTrainingExperiment(TrainingExperiment):
    train_subject_ids: list[int]
    test_subject_ids: list[int]
    train_reps: list[int]
    validate_reps: list[int]
    test_reps: list[int]

    training_data: list[str] = field(default_factory=list)    
    training_subjects_test_result: dict = field(default=None)
    test_subjects_test_result: dict = field(default=None)


    model_type: str = field(init=False, default='NN')
    experiment_type: str = field(init=False, default='other-subjects')

    @classmethod
    def create(cls, 
               train_subject_ids: list[int], 
               test_subject_ids: list[int], 
               train_reps: list[int], 
               validate_reps: list[int], 
               test_reps: list[int]) -> "NeuralNetworkOtherSubjectsTrainingExperiment":
        
        id_data = {
            "model_type": cls.model_type,
            "experiment_type": cls.experiment_type,
            "train_subject_ids": json.dumps([int(subject_id) for subject_id in train_subject_ids]),
            "test_subject_ids": json.dumps([int(subject_id) for subject_id in test_subject_ids]),
            "train_reps": json.dumps(train_reps),
            "validate_reps": json.dumps(validate_reps),
            "test_reps": json.dumps(test_reps)
        }
        id = ".".join(f"{k}:{v}" for k, v in id_data.items())

        return cls(
            id=id,
            train_subject_ids=train_subject_ids,
            test_subject_ids=test_subject_ids, 
            train_reps=train_reps, 
            validate_reps=validate_reps, 
            test_reps=test_reps
        )

    def save_training_data(self, epoch_training_data: dict): 
        self.training_data.append(json.dumps(epoch_training_data))

    def save_test_result(self, train_subjects_classification_report: dict, test_subjects_classification_report: dict): 
        self.test_result = {
            "train_subjects": {
                "f1_score": train_subjects_classification_report['macro avg']['f1-score'],
                "report": json.dumps(train_subjects_classification_report)
            },
            "test_subjects": {
                "f1_score": test_subjects_classification_report['macro avg']['f1-score'],
                "report": json.dumps(test_subjects_classification_report)
            },
        }

    def to_json_dict(self):
        return {
            "id": self.id,     
            "processing_duration": self.processing_duration,   
            "training_data": self.training_data,
            "test_result": self.test_result
        }
    
    @classmethod
    def _can_create_from_json_dict(cls, json_dict: dict):
        id = json_dict.get("id", "")
        return f"model_type:{cls.model_type}" in id and f"experiment_type:{cls.experiment_type}" in id 

    @classmethod
    def _from_json_dict(cls, json_dict: dict):
        
        id_data = dict(pair.split(":") for pair in json_dict["id"].split("."))

        instance = cls(
            id=json_dict["id"],
            train_subject_ids=list(json.loads(id_data["train_subject_ids"])), 
            test_subject_ids=list(json.loads(id_data["test_subject_ids"])), 
            train_reps=list(json.loads(id_data["train_reps"])), 
            validate_reps=list(json.loads(id_data["validate_reps"])), 
            test_reps=list(json.loads(id_data["test_reps"])), 
            training_data=json_dict.get("training_data", []),
            training_subjects_test_result=get_nested(value=json_dict, key_path="test_result:train_subjects", default=None),
            test_subjects_test_result=get_nested(value=json_dict, key_path="test_result:test_subjects", default=None)
        )
        instance.set_processing_duration(json_dict.get("processing_duration", None))
        return instance
    

@TrainingExperiment.register
@dataclass(eq=False)
class NeuralNetworkFineTunedTrainigExperiment(TrainingExperiment):
    subject_id: int
    pre_train_experiment_id: str
    train_reps: list[int]
    validate_reps: list[int]
    test_reps: list[int]
    training_data: list[str] = field(default_factory=list)
    test_result: dict = field(default=None)

    model_type: str = field(init=False, default='NN')
    experiment_type: str = field(init=False, default='fine-tuned')

    @classmethod
    def create(cls, 
               subject_id: int, 
               pre_train_experiment_id: str,
               train_reps: list[int], validate_reps: list[int], test_reps: list[int]) -> "NeuralNetworkSingleSubjectTrainingExperiment":
        
        id_data = {
            "model_type": cls.model_type,
            "experiment_type": cls.experiment_type,
            "subject_id": subject_id,
            "train_reps": json.dumps(train_reps),
            "validate_reps": json.dumps(validate_reps),
            "test_reps": json.dumps(test_reps)
        }
        id = ".".join(f"{k}:{v}" for k, v in id_data.items())

        return cls(
            id=id,
            pre_train_experiment_id=pre_train_experiment_id,
            subject_id=subject_id, 
            train_reps=train_reps, 
            validate_reps=validate_reps, 
            test_reps=test_reps
        )

    def save_training_data(self, epoch_training_data: dict): 
        self.training_data.append(json.dumps(epoch_training_data))

    def save_test_result(self, classification_report: dict): 
        self.test_result = {
            "f1_score": classification_report['macro avg']['f1-score'],
            "report": json.dumps(classification_report)
        }

    def to_json_dict(self):
        return {
            "id": self.id,
            "pre_train_experiment_id": self.pre_train_experiment_id,
            "processing_duration": self.processing_duration, 
            "training_data": self.training_data,
            "test_result": self.test_result
        }
    
    @classmethod
    def _can_create_from_json_dict(cls, json_dict: dict):
        id = json_dict.get("id", "")
        return f"model_type:{cls.model_type}" in id and f"experiment_type:{cls.experiment_type}" in id 

    @classmethod
    def _from_json_dict(cls, json_dict: dict):
        
        id_data = dict(pair.split(":") for pair in json_dict["id"].split("."))

        instance = cls(
            id=json_dict["id"],
            pre_train_experiment_id=json_dict["pre_train_experiment_id"],
            subject_id=int(id_data["subject_id"]), 
            train_reps=list(json.loads(id_data["train_reps"])), 
            validate_reps=list(json.loads(id_data["validate_reps"])), 
            test_reps=list(json.loads(id_data["test_reps"])), 
            training_data=json_dict.get("training_data", []),
            test_result=json_dict.get("test_result", None)
        )
        instance.set_processing_duration(json_dict.get("processing_duration", None))
        return instance