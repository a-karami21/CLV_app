import streamlit as st
import pandas as pd

# Page Config
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

col1 = st.sidebar
col2, col3 = st.columns((2,1))

# Overview Section

# Main Panel Title
st.title('Install Base Matrix')
st.write("This page shows customers and its install bases")

with st.container():
    with col1.container():
        # Sidebar Title
        col1.header("1. Install Base Data")
        with col1.expander("Dataset", expanded=True):
            # Uploader
            install_base_file = st.file_uploader('Upload the install base data')

if install_base_file is not None:
    @st.experimental_memo
    def read_install_base_csv():
        install_base = pd.read_csv(install_base_file, sep = ",")
        return install_base

    install_base = read_install_base_csv()
    install_base_edit = install_base
    #install_base_edit.drop(['Overall Rank', 'Segment Rank'], axis=1)
    install_base_edit = install_base_edit.set_index('Customer Name')

    edited_install_base = st.experimental_data_editor(install_base_edit)
