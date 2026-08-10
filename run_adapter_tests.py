import os
import sys
import pytest

sys.path.insert(0, os.path.abspath('.'))
sys.exit(pytest.main(['tests/ml/data']))
