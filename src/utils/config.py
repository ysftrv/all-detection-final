"""Merkezi config.yaml yükleyici — tüm modüller buradan import eder."""

import os
import yaml


def load_config(path: str = "config.yaml") -> dict:
    """config.yaml'ı yükle ve dict olarak döndür."""
    if not os.path.isfile(path):
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(root, "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
