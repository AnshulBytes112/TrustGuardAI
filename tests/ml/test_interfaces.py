import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from ml.interfaces import DatasetAdapter


def test_interface():
    adapter = DatasetAdapter()
    assert hasattr(adapter, "load")
