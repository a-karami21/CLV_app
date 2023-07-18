from PIL import Image

from src.lib.preparations import *
from src.lib.util import *

# Page Config

st.set_page_config(page_title="Homepage", page_icon="📒", layout="wide", initial_sidebar_state="expanded")

# Side Bar & Main Panel Setup
col1 = st.sidebar
col2, col3 = st.columns((2, 1))

# Initialization

# Load config
config = load_config("config_streamlit.toml")
ss = st.session_state

# initialize session state variable for modelling
if "df0" not in ss:
    ss.df0 = None
if "ss.df_columns" not in ss:
    ss.df_columns = None
if "product_list" not in ss:
    ss.product_list = None
if "df_list" not in ss:
    ss.df_list = None
if "df_filtered_list" not in ss:
    ss.df_filtered_list = None
if "df_ch_list" not in ss:
    ss.df_ch_list = None
if "df_rft_list" not in ss:
    ss.df_rft_list = None
if "bgf_list" not in ss:
    ss.bgf_list = None
if "bgf_eval_list_predicted" not in ss:
    ss.bgf_eval_list_predicted = None
if "bgf_eval_list_actual" not in ss:
    ss.bgf_eval_list_actual = None
if "bgf_full_list" not in ss:
    ss.bgf_full_list = None
if "ggf_list" not in ss:
    ss.ggf_list = None
if "df_rftv_list" not in ss:
    ss.df_rftv_list = None
if "df_viz_list" not in ss:
    ss.df_viz_list = None
if "merged_df" not in ss:
    ss.merged_df = None

# App Title & Description
with st.container():
    app_title = config["app"]["app_title"]
    st.title(app_title)

    dataset_file = config["app"]["app_dataset"]
    annual_discount_rate = 0.06

# Data Loading
if dataset_file is not None and ss.df0 is None:
    ss.df0 = read_order_intake_csv(dataset_file)

# Main Panel Section
st.divider()
st.header("1. Overview")
# App Explanation & Guide
with st.expander("App Overview", expanded=True):

    # App Description
    app_description = config["app"]["app_description"]
    st.markdown(app_description)

    left_column, right_column = st.columns(2)
    # App Purpose
    with left_column:
        app_overview_definition = config["app"]["app_overview_definition"]
        st.markdown(app_overview_definition)

    # App User Guide
    with right_column:
        app_overview_guide = config["app"]["app_overview_guide"]
        st.markdown(app_overview_guide)

with st.container():
    left_column, right_column = st.columns(2)
    with left_column:
        # App Workflow
        with st.expander("App Conceptual Model"):
            st.markdown("**Conceptual Model**")
            app_conceptual_model_image = config["app"]["app_conceptual_model_image"]
            image = Image.open(app_conceptual_model_image)
            st.image(image, use_column_width="auto")

    with right_column:
        # View Raw Dataset
        if ss.df0 is not None:
            with st.expander("View Dataset"):
                st.markdown("Sample Customer Transaction Dataset from April 2018 to March 2022.")
                st.dataframe(ss.df0)