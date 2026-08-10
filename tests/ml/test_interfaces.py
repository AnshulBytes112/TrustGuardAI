
from ml.interfaces import DatasetAdapter


def test_interface():
    adapter = DatasetAdapter()
    assert hasattr(adapter, "load")
