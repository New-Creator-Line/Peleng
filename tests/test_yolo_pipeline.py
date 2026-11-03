import os
from pathlib import Path
import pytest
from train_yolo import TrainingConfig, validate_environment, set_seed


def test_training_config_defaults():
    cfg = TrainingConfig()
    assert cfg.model_name == "yolov8x.pt"
    assert cfg.data_yaml == "peleng.v1i.yolov8/data.yaml"
    assert cfg.imgsz == 512
    assert cfg.batch_size == 8
    assert cfg.patience == 15
    assert cfg.seed == 42


def test_validate_environment_success():
    cfg = TrainingConfig(data_yaml="peleng.v1i.yolov8/data.yaml")
    # Should not raise
    validate_environment(cfg)


def test_validate_environment_missing_file():
    cfg = TrainingConfig(data_yaml="non_existent_data.yaml")
    with pytest.raises(FileNotFoundError):
        validate_environment(cfg)


def test_reproducibility_seed():
    set_seed(123)
    import random
    import numpy as np
    import torch

    r1 = random.random()
    n1 = np.random.rand()
    t1 = torch.rand(1).item()

    set_seed(123)
    r2 = random.random()
    n2 = np.random.rand()
    t2 = torch.rand(1).item()

    assert r1 == r2
    assert n1 == n2
    assert t1 == t2
