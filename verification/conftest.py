import os
import sys

# make the repo-root modules (surface_code, noreset_tools, ...) importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
