import os
from pathlib import Path
import pytest
from scripts.training.train_rtdetr import RTDETRConfig, validate_environment, set_seed


def test_rtdetr_config_defaults():
    cfg = RTDETRConfig()
    assert cfg.model_name == "weights/pretrained/rtdetr-l.pt"
    assert cfg.data_yaml == "data/datasets/yolo_v3/data.yaml"
    assert cfg.imgsz == 640
    assert cfg.batch_size == 4
    assert cfg.patience == 15
    assert cfg.seed == 42
    assert cfg.erasing == 0.4
    assert cfg.best_weight_name == "peleng_rtdetr_best.pt"


def test_rtdetr_validate_environment_success():
    yaml_path = Path("data/datasets/yolo_v3/data.yaml")
    if not yaml_path.exists():
        pytest.skip("Training dataset not present in repository")
    cfg = RTDETRConfig(data_yaml=str(yaml_path))
    validate_environment(cfg)


def test_rtdetr_validate_environment_missing_file():
    cfg = RTDETRConfig(data_yaml="non_existent_data.yaml")
    with pytest.raises(FileNotFoundError):
        validate_environment(cfg)


def test_rtdetr_reproducibility_seed():
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
