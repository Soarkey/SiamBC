from __future__ import absolute_import, division, print_function

import os.path as osp
import sys

lib_path = osp.join(osp.dirname(__file__), '../..', 'eval_toolkit')
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)
