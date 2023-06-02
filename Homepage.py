import streamlit as st
import configparser

from PIL import Image

from lifetimes.plotting import \
    plot_period_transactions, \
    plot_calibration_purchases_vs_holdout_purchases

from scipy.stats import chi2_contingency
from scipy.stats import chi2

from src.lib.preparations import *
from src.lib.visualizations import *
from src.lib.util import *

## Streamlit Setup
# Page Config

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# Side Bar & Main Panel Setup
col1 = st.sidebar
col2, col3 = st.columns((2,1))

# Initialization

# Load config
config = load_config("config_streamlit.toml")

ss = st.session_state

# initialize session state variable for modelling
if "df0" not in ss:
    ss.df0 = None
if "bu_list" not in ss:
    ss.bu_list = None
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
if "bgf_full_list" not in ss:
    ss.bgf_full_list = None
if "ggf_list" not in ss:
    ss.ggf_list = None
if "df_rftv_list" not in ss:
    ss.df_rftv_list = None
if "df_viz_list" not in ss:
    ss.df_viz_list = None

# initialize session state variable for user input
if "customer_id_input" not in ss:
    ss.customer_id_input = None
if "customer_name_input" not in ss:
    ss.customer_name_input = None
if "prob_alive_input" not in ss:
    ss.prob_alive_input = None
if "expected_lifetime" not in ss:
    ss.expected_lifetime = None

# App Title & Description
with st.container():
    app_title = config["app"]["app_title"]
    st.title(app_title)

    app_description = config["app"]["app_description"]
    st.markdown(app_description)

# Overview Section
st.header("1. Overview")

# App Explanation & Guide
with st.expander("App Overview"):
    left_column, right_column = st.columns(2)
    # App Explanation
    with left_column:
        app_overview_definition = config["app"]["app_overview_definition"]
        st.markdown(app_overview_definition)

    # App User Guide
    with right_column:
        app_overview_guide = config["app"]["app_overview_guide"]
        st.markdown(app_overview_guide)

# App Workflow
with st.expander("App Workflow"):
    st.markdown("**Workflow**")
    app_workflow_image = config["app"]["app_workflow_image"]
    image = Image.open(app_workflow_image)
    st.image(image, caption='Basic Workflow', width=800)

# Side Bar Inputs
with col1.container():
    # Sidebar Title
    col1.header("1. Data")
    with col1.expander("Dataset", expanded=True):
        # Uploader
        orderintake_file = st.file_uploader('Upload the order intake data')

        # Sidebar Industry Options
        industry_type_selection = st.selectbox('Select industry type', ('Energy', 'Material', 'Life', 'All'), index=3)

    col1.header("2. Prediction Inputs")
    with col1.expander("User Inputs"):
        # Predict Purchase Range Number
        annual_discount_rate = st.number_input("Input Annual Interest Rate (Default is 6%)", 0.01, 0.25, value=0.06)
        expected_lifetime = st.slider("Select prediction lifetime (Months)", 12, 120, step=12)

        ss.expected_lifetime = expected_lifetime

# Data Loading
if orderintake_file is not None and ss.df0 is None:
    ss.df0 = read_order_intake_csv(orderintake_file)

if ss.df0 is not None:
    # Show Data Info
    with st.container():
        # treat CustomerID as a categorical variable
        ss.df0["ec_eu_customer"] = ss.df0["ec_eu_customer"].astype(np.int64).astype(object)

        # Filter to business unit input
        ss.df0 = ss.df0[ss.df0['product_type'].isin(["Product"])]
        ss.df0 = industry_filter(industry_type_selection, ss.df0)

        ss.bu_list = ["YY113", "YY116", "YY117", "YY119"]

        ss.df_list = {}
        for bu in ss.bu_list:
            df_bu = ss.df0[ss.df0['BU'].isin([bu])]
            ss.df_list[bu] = df_bu

    # Model Evaluation Section
    st.header("2. Model Evaluation")

    # Checkbox to trigger training
    train_checkbox = st.checkbox("Train Model")

    # Train BG/NBD to Fit on the full dataset
    with st.container():
        if train_checkbox:
            ## BG/NBD Modelling

            # Train test split
            ss.df_ch_list = {}
            for bu, df in ss.df_list.items():
                df_ch = train_test_split(df)
                ss.df_ch_list[bu] = df_ch

            # Finding max date
            ss.max_date_list = {}
            for bu, df in ss.df_list.items():
                max_date = max_date_func(df)
                ss.max_date_list[bu] = max_date

            # RFM aggregation dataframe
            ss.df_rft_list = {}
            for (bu, df), (bu, max_date) in zip(ss.df_list.items(), ss.max_date_list.items()):
                df_rft = determine_df_rft(df, max_date)
                ss.df_rft_list[bu] = df_rft

            # Fitting BG/NBD
            ss.bgf_list = {}
            for bu, df_ch in ss.df_ch_list.items():
                bgf = bgf_fit(df_ch)
                ss.bgf_list[bu] = bgf

            ## Model Evaluation
            # Predicted Frequency
            bgf_eval_list_predicted = {}
            for (bu, bgf), (bu, df_ch) in zip(ss.bgf_list.items(), ss.df_ch_list.items()):
                bgf_eval = bgf_eval_prep_predicted(bgf, df_ch)
                bgf_eval_list_predicted[bu] = bgf_eval

            # Actual Frequency
            bgf_eval_list_actual = {}
            for (bu, bgf), (bu, df_ch) in zip(ss.bgf_list.items(), ss.df_ch_list.items()):
                bgf_eval = bgf_eval_prep_actual(bgf, df_ch)
                bgf_eval_list_actual[bu] = bgf_eval

            # Train model to the full dataset
            ss.bgf_full_list = {}
            for bu, df_rft in ss.df_rft_list.items():
                bgf_full = bgf_full_fit(df_rft)
                ss.bgf_full_list[bu] = bgf_full

            ## Predicting future purchase
            # call helper function: predict each customer's purchases over multiple time periods
            t_FC = [90, 180, 270, 360]

            for t in t_FC:
                for (bu, df_rft), (bu, bgf) in zip(ss.df_rft_list.items(), ss.bgf_list.items()):
                    predict_purch(df_rft, bgf, t)

            # Calculate probability alive
            for (bu,bgf), (bu, df_rft) in zip(ss.bgf_full_list.items(), ss.df_rft_list.items()):
                df_rft["prob_alive"] = calculate_prob_alive(bgf, df_rft)

            # Create new dataframe (added predicted future purchase & prob alive)
            ss.df_rftv_list = {}
            for bu, df_rft in ss.df_rft_list.items():
                df_rftv = gg_prep(df_rft)
                ss.df_rftv_list[bu] = df_rftv

            ## Gamma-Gamma Modelling

            # Calculate F & M Pearson Correlation
            ss.corr_list = {}
            for bu, df_rftv in ss.df_rftv_list.items():
                corr = gg_corr(df_rftv)
                ss.corr_list[bu] = corr

            # Gamma-Gamma Model Fitting
            ss.ggf_list = {}
            for bu, df_rftv in ss.df_rftv_list.items():
                ggf = gg_fit(df_rftv)
                ss.ggf_list[bu] = ggf

            # Estimate average error of the predicted monetary value
            for (bu, df_rftv), (bu, ggf) in zip(ss.df_rftv_list.items(), ss.ggf_list.items()):
                df_rftv_new = gg_avg(df_rftv, ggf)
                ss.df_rftv_list[bu] = df_rftv_new

            # Calculate MAPE for Gamma-Gamma evaluation
            ss.mape_list = {}
            for bu, df_rftv in ss.df_rftv_list.items():
                mape = gg_evaluation(df_rftv)
                ss.mape_list[bu] = mape

            # Predict Customer Lifetime Value and add to the output dataframe
            for (bu, df_rftv), (bu, ggf), (bu, bgf) in zip(ss.df_rftv_list.items(), ss.ggf_list.items(), ss.bgf_full_list.items()):
                df_rftv_new2 = compute_clv(df_rftv, ggf, bgf, annual_discount_rate, ss.expected_lifetime)
                ss.df_rftv_list[bu] = df_rftv_new2

            ## Model Evaluation Visualization

            # Predicted vs Actual (Train-Test) Graph Container
            with st.container():
                st.write("Training BG/NBD Model: does the model reflect the actual data closely enough?")
                left_column, right_column = st.columns(2)
                # Predicted vs Actual Chart By Number of Customer
                with left_column:
                    tab_list = st.tabs(ss.bu_list)
                    for bu, tab in zip(ss.bu_list, tab_list):
                        with tab:
                            fig1 = eva_viz1(ss.bgf_list[bu])
                            st.pyplot(fig1.figure)

                # Predicted vs Actual By Holdout & Calibration Data Purchases
                with right_column:
                    tab_list = st.tabs(ss.bu_list)
                    for bu, tab in zip(ss.bu_list, tab_list):
                        with tab:
                            fig2 = plot_calibration_purchases_vs_holdout_purchases(ss.bgf_list[bu],
                                                                                   ss.df_ch_list[bu], n=9)
                            st.pyplot(fig2.figure)

            # Model Performance Metrics
            with st.container():
                left_column, middle_column, right_column = st.columns(3)

                # RMSE, MAE, Pearson Correlation, MAPE
                with left_column:
                    with st.expander("Performance Score"):
                        tab_list = st.tabs(ss.bu_list)
                        for bu, tab in zip(ss.bu_list, tab_list):
                            with tab:
                                st.write("BG/NBD Model Performance:")
                                st.markdown("* MAE: {0}".format(score_model(bgf_eval_list_predicted[bu],
                                                                            bgf_eval_list_actual[bu],
                                                                            'mae')))
                                st.markdown("* RMSE: {0}".format(score_model(bgf_eval_list_predicted[bu],
                                                                            bgf_eval_list_actual[bu],
                                                                             'rmse')))
                                st.write("Gamma-Gamma Model Performance:")
                                st.markdown("* Pearson correlation: %.3f" % ss.corr_list[bu])
                                st.markdown("* MAPE of predicted revenues: " + f'{ss.mape_list[bu]:.2f}')

                # Chi-Square Test (Customer Count)
                with middle_column:
                    with st.expander("Prediction Difference of Customer Count"):
                        tab_list = st.tabs(ss.bu_list)
                        for bu, tab in zip(ss.bu_list, tab_list):
                            with tab:
                                df_chi_square_test_customer_count, chi, pval,\
                                dof, exp, significance, critical_value = chi_square_test_customer_count(bu, ss.bgf_list)

                                st._legacy_dataframe(df_chi_square_test_customer_count.style.format("{:,.0f}"))

                                st.markdown('p-value is {:.5f}'.format(pval))
                                st.markdown('chi = %.6f, critical value = %.6f' % (chi, critical_value))
                                if chi > critical_value:
                                    st.markdown("""At %.3f level of significance, we reject the null hypotheses and accept H1.
                                  There is significant difference between actual and model.""" % (significance))
                                else:
                                    st.markdown("""At %.3f level of significance, we accept the null hypotheses.
                                  There is no significant difference between actual and model.""" % (significance))

                # Chi-Square Test (Holdout vs Calibration Purchases)
                with right_column:
                    with st.expander("Prediction Difference of Calibration vs Holdout Purchases"):
                        tab_list = st.tabs(ss.bu_list)
                        for bu, tab in zip(ss.bu_list, tab_list):
                            with tab:
                                df_chi_square_test_cal_vs_hol, chi, pval, \
                                dof, exp, significance, critical_value = chi_square_test_cal_vs_hol(ss.df_ch_list, ss.bgf_list, bu)

                                st._legacy_dataframe(df_chi_square_test_cal_vs_hol.style.format("{:,.2f}"))

                                st.markdown('p-value is {:.5f}'.format(pval))

                                st.markdown('chi = %.6f, critical value = %.6f' % (chi, critical_value))

                                if chi > critical_value:
                                    st.markdown("""At %.3f level of significance, we reject the null hypotheses and accept H1.
                                There is significant difference between actual and model.""" % (significance))
                                else:
                                    st.markdown("""At %.3f level of significance, we accept the null hypotheses.
                                There is no significant difference between actual and model.""" % (significance))

            with st.expander("Show Result"):
                if ss.df_rftv_list is not None:
                    # Display full table from the result
                    st.write("Full Table")

                    ss.df_viz_list = {}
                    for (bu, df_final), (bu, df) in zip(ss.df_rftv_list.items(), ss.df_list.items()):
                        df_viz = export_table(df_final, df)
                        ss.df_viz_list[bu] = df_viz

                    tab_list = st.tabs(ss.bu_list)
                    for bu, tab in zip(ss.bu_list, tab_list):
                        with tab:

                            st.dataframe(ss.df_viz_list[bu].style.format({
                                "CLV": "{:,.2f}",
                                "frequency": "{:,.0f}",
                                "recency": "{:,.0f}",
                                "T": "{:,.0f}",
                                "monetary_value": "{:,.2f}",
                                "predict_purch_90": "{:,.2f}",
                                "predict_purch_180": "{:,.2f}",
                                "predict_purch_270": "{:,.2f}",
                                "predict_purch_360": "{:,.2f}",
                                "prob_alive": "{:,.2f}",
                                "exp_avg_rev": "{:,.2f}",
                                "avg_rev": "{:,.2f}",
                                "error_rev": "{:,.2f}",
                            }))

                            df_csv = convert_df(ss.df_viz_list[bu])

                            st.download_button("Press to Download",
                                               df_csv,
                                               "CLV Table.csv",
                                               "text/csv",
                                               key='download-csv-product'+ bu
                                               )

                            # Display full table from the result
                            st.write("Result Stats")

                            st.dataframe(ss.df_viz_list[bu].describe().style.format("{:,.2f}"))
