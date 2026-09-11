"""Entry point for hosting: runs Code/dashboard.py.

Streamlit Community Cloud is pointed at this file. The application
itself lives in Code/; this file only puts Code/ on the import path and
executes it, so the hosted app keeps working without a redeploy.

    python -m streamlit run dashboard.py        # same as Code/dashboard.py
"""
import os
import runpy
import sys

_CODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Code")
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)
runpy.run_path(os.path.join(_CODE, "dashboard.py"), run_name="__main__")
