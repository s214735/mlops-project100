import os
import sys

_TEST_ROOT = os.path.dirname(__file__)  # root of test folder
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)  # root of project
_PATH_DATA = os.path.join(_PROJECT_ROOT, "data/processed")  # root of data

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/')))