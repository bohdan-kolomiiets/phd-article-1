import uuid
import numpy as np
from libemg_3dc.utils.training_experiments import TrainingExperiment, NeuralNetworkSingleSubjectTrainingExperiment

def test_single_subject_from_json_and_back():
    test_id = str(uuid.uuid4())
    json_dict = {
        "id": test_id,
        "model_type": "NN",
        "experiment_type": "single-subject",
        "subject_id": "42",
        "repetitions-split": {
            "training-reps": [1, 2, 3],
            "validation-reps": [4],
            "test-reps": [5]
        }
        # "model-filename": f"{test_id}.pt",
        # "tensorboard-directory": test_id
    }

    obj = TrainingExperiment.from_json(json_dict)
    assert isinstance(obj, NeuralNetworkSingleSubjectTrainingExperiment)
    assert obj.subject_id == "42"
    assert (obj.train_reps == np.array([1, 2, 3])).all()

    roundtrip_json = obj.to_json_dict()
    assert roundtrip_json["id"] == test_id
    assert roundtrip_json["repetitions-split"]["test-reps"] == [5]