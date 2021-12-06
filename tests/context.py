""" Factorize path manipulation out to this file.
Each test file just imports context and then can proceed as normal.
"""

import os
from os.path import dirname, join, abspath
import sys

# import the path to the DNN-SE package
# (this is just adding the parent directory of `context.py` to the sys path)
sys.path.insert(0, abspath(join(dirname(__file__), '..')))