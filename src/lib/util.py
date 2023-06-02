from typing import Any, Dict, Tuple

from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import toml

def get_project_root() -> str:
    """Returns project root path.

    Returns
    -------
    str
        Project root path.
    """
    return str(Path(__file__).parent.parent.parent)

@st.cache(allow_output_mutation=True, ttl=300)
def load_config(config_streamlit_filename):

    config_streamlit = toml.load(Path(get_project_root()) / f"config/{config_streamlit_filename}")

    return dict(config_streamlit)