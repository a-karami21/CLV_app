import numpy as np
import streamlit as st
import configparser

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from bokeh.plotting import figure
from bokeh.palettes import GnBu6, OrRd6
import altair as alt

import math

from lifetimes.plotting import \
    plot_period_transactions, \
    plot_calibration_purchases_vs_holdout_purchases, \
    plot_history_alive

from PIL import Image

from scipy.stats import chi2_contingency
from scipy.stats import chi2

from src.lib.preparations import *
from src.lib.visualizations import *
from src.lib.util import *

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

# initialize session state variable for user input
if "selected_columns_dict" not in ss:
    ss.selected_columns_dict = None
if "attribute_is_valid" not in ss:
    ss.attribute_is_valid = None
if "customer_id_input" not in ss:
    ss.customer_id_input = None
if "customer_name_input" not in ss:
    ss.customer_name_input = None
if "prob_alive_input" not in ss:
    ss.prob_alive_input = None

# App Title & Description
with st.container():
    app_title = config["app"]["app_title"]
    st.title(app_title)

    app_description = config["app"]["app_description"]
    st.markdown(app_description)

# Sidebar Options
# Data Upload
col1.header("1. Data")
with col1.expander("Dataset", expanded=True):
    # Uploader
    dataset_file = st.file_uploader('Upload the order intake data')

    # Annual Discount Rate Selection
    annual_discount_rate = st.number_input("Input Annual Interest Rate (Default is 6%) for CLV Prediction",
                                           min_value=0.01,
                                           max_value=0.25,
                                           value=0.06)

# Data Loading
if dataset_file is not None and ss.df0 is None:
    ss.df0 = read_order_intake_csv(dataset_file)

# Modelling Setup
if ss.df0 is not None:
    col1.header("2. Setup")
    with col1.expander("Attribute Selection", expanded=True):
        # Get Column Name of the Uploaded Dataframe
        ss.df_columns = ss.df0.columns.tolist()
        ss.df_columns.insert(0, "Not Selected")

        # User Attribute Selection
        with st.form("Attribute Selection"):
            df_date_col = st.selectbox("Select Date Attribute", ss.df_columns, key="Date")
            df_product_category_col = st.selectbox("Select Product Attribute", ss.df_columns, key="Product")
            df_customer_id_col = st.selectbox("Select Customer ID Attribute", ss.df_columns, key="Customer_ID")
            df_customer_name_col = st.selectbox("Select Customer Name Attribute", ss.df_columns, key="Customer_Name")
            df_industry_col = st.selectbox("Select Industry Attribute", ss.df_columns, key="Industry")
            df_transaction_value_col = st.selectbox("Select Monetary Attribute", ss.df_columns, key="Transaction_Value")

            submitted = st.form_submit_button("Submit")

            # Snapshot the selected attributes in a dictionary
            ss.selected_columns_dict = {"Date": df_date_col,
                                     "Product": df_product_category_col,
                                     "Customer_ID": df_customer_id_col,
                                     "Customer_Name": df_customer_name_col,
                                     "Industry": df_industry_col,
                                     "Transaction_Value": df_transaction_value_col}

            if submitted:
                for key, value in ss.selected_columns_dict.items():
                    if value == "Not Selected":
                        st.write(f"Please select the {key} attributes")
                        ss.attribute_is_valid = False
                        break
                    else:
                        ss.attribute_is_valid = True

# Main Panel Section

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


if ss.df0 is not None and ss.attribute_is_valid:
    # Model Evaluation Section
    st.header("2. Model Evaluation")

    # Checkbox to Trigger Modelling Functions
    train_checkbox = st.checkbox("Train Model")

    # Data Preparation
    ss.product_list, ss.df_list = modelling_data_prep(ss.df0)

    # Train BG/NBD to Fit on the full dataset
    if train_checkbox:
        with st.container():
            # BG/NBD Modelling
            #   Train test split
            ss.df_ch_list = {}
            for product, df in ss.df_list.items():
                df_ch = train_test_split(df)
                ss.df_ch_list[product] = df_ch

            #   Finding max date
            ss.max_date_list = {}
            for product, df in ss.df_list.items():
                max_date = max_date_func(df)
                ss.max_date_list[product] = max_date

            #   RFM aggregation dataframe
            ss.df_rft_list = {}
            for (product, df), (product, max_date) in zip(ss.df_list.items(), ss.max_date_list.items()):
                df_rft = determine_df_rft(df, max_date)
                ss.df_rft_list[product] = df_rft

            #   Fitting BG/NBD
            ss.bgf_list = {}
            for product, df_ch in ss.df_ch_list.items():
                bgf = bgf_fit(df_ch)
                ss.bgf_list[product] = bgf

            # Model Evaluation
            #   Predicted Frequency
            bgf_eval_list_predicted = {}
            for (product, bgf), (product, df_ch) in zip(ss.bgf_list.items(), ss.df_ch_list.items()):
                bgf_eval = bgf_eval_prep_predicted(bgf, df_ch)
                bgf_eval_list_predicted[product] = bgf_eval

            #   Actual Frequency
            bgf_eval_list_actual = {}
            for (product, bgf), (product, df_ch) in zip(ss.bgf_list.items(), ss.df_ch_list.items()):
                bgf_eval = bgf_eval_prep_actual(bgf, df_ch)
                bgf_eval_list_actual[product] = bgf_eval

            #   Train model to the full dataset
            ss.bgf_full_list = {}
            for product, df_rft in ss.df_rft_list.items():
                bgf_full = bgf_full_fit(df_rft)
                ss.bgf_full_list[product] = bgf_full

            #   Predicting future purchase
            #   call helper function: predict each customer's purchases over multiple time periods
            t_FC = [90, 180, 270, 360]

            for t in t_FC:
                for (product, df_rft), (product, bgf) in zip(ss.df_rft_list.items(), ss.bgf_list.items()):
                    predict_purch(df_rft, bgf, t)

            #   Calculate probability alive
            for (product,bgf), (product, df_rft) in zip(ss.bgf_full_list.items(), ss.df_rft_list.items()):
                df_rft["prob_alive"] = calculate_prob_alive(bgf, df_rft)

            #   Create new dataframe (added predicted future purchase & prob alive)
            ss.df_rftv_list = {}
            for product, df_rft in ss.df_rft_list.items():
                df_rftv = gg_prep(df_rft)
                ss.df_rftv_list[product] = df_rftv

            # Gamma-Gamma Modelling

            #   Calculate F & M Pearson Correlation
            ss.corr_list = {}
            for product, df_rftv in ss.df_rftv_list.items():
                corr = gg_corr(df_rftv)
                ss.corr_list[product] = corr

            #   Gamma-Gamma Model Fitting
            ss.ggf_list = {}
            for product, df_rftv in ss.df_rftv_list.items():
                ggf = gg_fit(df_rftv)
                ss.ggf_list[product] = ggf

            #   Estimate average error of the predicted monetary value
            for (product, df_rftv), (product, ggf) in zip(ss.df_rftv_list.items(), ss.ggf_list.items()):
                df_rftv_new = gg_avg(df_rftv, ggf)
                ss.df_rftv_list[product] = df_rftv_new

            #   Calculate MAPE for Gamma-Gamma evaluation
            ss.mape_list = {}
            for product, df_rftv in ss.df_rftv_list.items():
                mape = gg_evaluation(df_rftv)
                ss.mape_list[product] = mape

            #   Predict Customer Lifetime Value and add to the output dataframe
            for (product, df_rftv), (product, ggf), (product, bgf) in zip(ss.df_rftv_list.items(), ss.ggf_list.items(), ss.bgf_full_list.items()):
                df_rftv_new2 = compute_clv(df_rftv, ggf, bgf, annual_discount_rate)
                ss.df_rftv_list[product] = df_rftv_new2

        # Model Evaluation Visualization
        with st.expander("Model Evaluation Result"):
            st.subheader("Model vs Actual")
            tab_list = st.tabs(ss.product_list)
            for product, tab in zip(ss.product_list, tab_list):
                with tab:
                    # Predicted vs Actual (Train-Test) Graph Container
                    with st.container():
                        left_column, right_column = st.columns(2)

                        evaluation_plot_dict = evaluation_visualization(ss.bgf_list[product], ss.df_ch_list[product])

                        # Predicted vs Actual Chart By Number of Customer
                        with left_column:
                            st.pyplot(evaluation_plot_dict["Plot1"].figure)

                        # Predicted vs Actual By Holdout & Calibration Data Purchases
                        with right_column:
                            st.pyplot(evaluation_plot_dict["Plot2"].figure)

                    # UI Divider
                    st.divider()

                    # Model Performance Metrics
                    with st.container():
                        left_column, middle_column, right_column = st.columns(3)

                        # RMSE, MAE, Pearson Correlation, MAPE
                        with left_column:
                            st.write("BG/NBD Model Performance:")
                            st.markdown("* MAE: {0}".format(score_model(bgf_eval_list_predicted[product],
                                                                        bgf_eval_list_actual[product],
                                                                        'mae')))
                            st.markdown("* RMSE: {0}".format(score_model(bgf_eval_list_predicted[product],
                                                                        bgf_eval_list_actual[product],
                                                                         'rmse')))
                            st.write("Gamma-Gamma Model Performance:")
                            st.markdown("* Pearson correlation: %.3f" % ss.corr_list[product])
                            st.markdown("* MAPE of predicted revenues: " + f'{ss.mape_list[product]:.2f}')

                        # Chi-Square Test (Customer Count)
                        with middle_column:
                            st.write("Prediction Difference of Customer Count")

                            df_chi_square_test_customer_count, chi, pval,\
                            dof, exp, significance, critical_value = chi_square_test_customer_count(product, ss.bgf_list)

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
                            st.write("Prediction Difference of Calibration vs Holdout Purchases")

                            df_chi_square_test_cal_vs_hol, chi, pval, \
                            dof, exp, significance, critical_value = chi_square_test_cal_vs_hol(ss.df_ch_list, ss.bgf_list, product)

                            st._legacy_dataframe(df_chi_square_test_cal_vs_hol.style.format("{:,.2f}"))

                            st.markdown('p-value is {:.5f}'.format(pval))
                            st.markdown('chi = %.6f, critical value = %.6f' % (chi, critical_value))

                            if chi > critical_value:
                                st.markdown("""At %.3f level of significance, we reject the null hypotheses and accept H1.
                            There is significant difference between actual and model.""" % (significance))
                            else:
                                st.markdown("""At %.3f level of significance, we accept the null hypotheses.
                            There is no significant difference between actual and model.""" % (significance))

        # Model Result (Table)
        with st.expander("Show Result"):
            if ss.df_rftv_list is not None:
                # Combining Model Result & Customer Identities
                ss.df_viz_list = {}
                for (product, df_final), (product, df) in zip(ss.df_rftv_list.items(), ss.df_list.items()):
                    ss.df_viz_list[product] = export_table(df_final, df)

                # Create a new dataframe to combine all product to one df
                with st.container():
                    # Create a new column named product Category based on the df product key
                    df_with_product_col = ss.df_viz_list
                    for product, df in df_with_product_col.items():
                        df["Product"] = product

                    # Combine All product df to one df
                    df_product_list = list(df_with_product_col.values())
                    ss.merged_df = df_product_list[0]
                    for df in df_product_list[1:]:
                        ss.merged_df = pd.concat([ss.merged_df, df], ignore_index=True)


                # Full Table & Stats Display
                tab_list = st.tabs(ss.product_list)
                for product, tab in zip(ss.product_list, tab_list):
                    with tab:
                        st.write("Full Table")
                        st.dataframe(ss.df_viz_list[product].style.format({
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

                        df_csv = convert_df(ss.merged_df)

                        st.download_button("Press to Download",
                                           df_csv,
                                           "CLV Table.csv",
                                           "text/csv",
                                           key='download-csv-product'+product
                                           )

                        # Display full table from the result
                        st.write("Result Stats")

                        st.dataframe(ss.df_viz_list[product].describe().style.format("{:,.2f}"))

if ss.df_viz_list is not None:
    st.header("3. Visualization")

    # RFM Spread Visualization
    with st.expander("RFM Stats", expanded=True):
        st.write("RFM Spread Visualization")
        tab_list = st.tabs(ss.product_list)
        for product, tab in zip(ss.product_list, tab_list):
            with tab:
                left_column, middle_column, right_column = st.columns(3)

                rfm_plot_dict = rfm_summary_figure(ss.df_rft_list, product)

                with left_column:
                    st.pyplot(rfm_plot_dict["Recency"].figure)
                with middle_column:
                    st.pyplot(rfm_plot_dict["Frequency"].figure)
                with right_column:
                    st.pyplot(rfm_plot_dict["Monetary"].figure)

    # Customer P Alive Plot
    with st.expander("Customer P(Alive) History Plot", expanded=True):
        tab_list = st.tabs(ss.product_list)
        for product, tab in zip(ss.product_list, tab_list):
            with tab:

                # Customer Selection
                customer_name_list = ss.df_viz_list[product]['Customer_Name']

                customer_selection = st.selectbox("Select customer ID to see its purchase behavior",
                                                  customer_name_list)

                selected_customer_id = ss.df_viz_list[product].loc[
                    ss.df_viz_list[product]["Customer_Name"] == customer_selection, 'Customer_ID'].values[0]
                selected_customer_name = ss.df_list[product][
                    ss.df_list[product]["Customer_ID"] == selected_customer_id]

                # P Alive History Plot & User Inputs
                with st.container():
                    p_alive_history_plot = fig_p_alive_history(product, ss.df_list, ss.bgf_full_list, selected_customer_name)
                    fig = st.pyplot(p_alive_history_plot.figure)

                # RFM & Last purchase date
                with st.container():
                    profile_col1, profile_col2, profile_col3, profile_col4, profile_col5, profile_col6 = st.columns(
                        6)

                    rfm_summary_dict = customer_rfm_summary(ss.df_viz_list, product, ss.df_list, selected_customer_id)

                    with profile_col1:
                        st.markdown("**Recency**")
                        st.markdown(str(rfm_summary_dict["Recency"]) + " Lifetime Length (Months)")
                    with profile_col2:
                        st.markdown("**Frequency**")
                        st.markdown(str(rfm_summary_dict["Frequency"]) + " Active Months")
                    with profile_col3:
                        st.markdown("**Monetary**")
                        st.markdown("$ " + f'{rfm_summary_dict["Monetary"]:,}' + " Average Purchase Value")
                    with profile_col4:
                        st.markdown("**Last Purchase Date**")
                        st.markdown(rfm_summary_dict["Last Purchase"])
                    with profile_col5:
                        st.markdown("**Predicted Purchase in Upcoming Year**")
                        st.markdown(str(rfm_summary_dict["Predicted Purchase"]) + " Purchases")
                    with profile_col6:
                        st.markdown("**CLV in Upcoming Year**")
                        st.markdown("$ " + f'{rfm_summary_dict["CLV"]:,}')

    # Top Customers
    with st.expander("CLV Visualizations", expanded=True):

        # Top 20 & Sales Growth Plot
        with st.container():
            df_plot_prep = df_plot_preparation(ss.df0, ss.merged_df)

            colors = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#EE8959", "#E76F51"]

            left_column, right_column = st.columns(2)
            with left_column:
                st.subheader("Top 20 Customers by CLV")

                industry_options = df_plot_prep["Product"].unique()
                industry_options.insert(0, "All")

                industry_filter_selection = st.selectbox("Industry Filter", industry_options, key="top20_industry")
                product_filter_selection = st.multiselect("Product Filter", df_plot_prep["Product"].unique(), key="top20_product")

                df_plot = top_cust_data_prep(df_plot_prep, industry_filter_selection, product_filter_selection)

                st.altair_chart(fig_top20(df_plot, colors), use_container_width=True)

            with right_column:
                st.subheader("Predicted Sales Growth")

                # Filter Selection
                with st.container():
                    left_column, right_column = st.columns(2)

                    # Industry & Product Filter
                    with left_column:

                        industry_options = df_plot_prep["Product"].unique()
                        industry_options.insert(0, "All")

                        industry_filter_selection = st.selectbox("Industry Filter", industry_options,
                                                                 key="growth_industry")
                        product_filter_selection = st.multiselect("Product Filter", df_plot_prep["Product"].unique(),
                                                                  key="growth_product")
                    # Customer Filter
                    with right_column:
                        predicted_customer_only = st.checkbox("Predicted Customer Only?", value=True)

                        if not predicted_customer_only:
                            df_customer_filter = df_plot_prep
                        else:
                            df_customer_filter = df_plot_prep[df_plot_prep['Customer_Name'].isin(
                                df_plot_prep[df_plot_prep['Fiscal_Year'] == 'FY2023(Predicted)']['Customer_Name'])]

                        customer_filter_list = df_customer_filter["Customer_Name"].unique().tolist()
                        customer_filter_list.insert(0, "All")
                        customer_filter_selection = st.selectbox("Customer Filter", customer_filter_list,
                                                                 key="growth_customer")

                        st.markdown("Note: The model only predicts customers that have repeated purchase"
                                    " (more than once) so the figure may be smaller than the previous year")


                df_plot = sales_growth_data_prep(df_plot_prep, industry_filter_selection,
                                                 product_filter_selection, customer_filter_selection)

                plot = fig_sales_growth(df_plot, colors)
                st.altair_chart(plot, use_container_width=True)

        # UI Divider
        st.divider()

        # Line Plot for Comparison
        with st.container():
            st.subheader("Comparison Line Plot")
            left_column, right_column = st.columns((1,4))

            # Line Plot Filters
            with left_column:
                line_plot_group = st.selectbox("Select Grouping Variable", ["Industry", "Industry_Segment", "Product"],
                                               index=0, key="line_plot_group")

                industry_filter_selection = st.multiselect("Industry Filter", df_plot_prep["Industry"].unique(),
                                                           key="line_plot_industry")

                if not industry_filter_selection:
                    df_industry_segment_selection = df_plot_prep
                else:
                    df_industry_segment_selection = df_plot_prep[df_plot_prep['Industry'].isin(industry_filter_selection)]

                industry_segment_filter_selection = st.multiselect("Industry Segment Filter",
                                                                   df_industry_segment_selection["Industry_Segment"].unique(),
                                                                   key="line_plot_industry_segment")

                product_filter_selection = st.multiselect("Product Filter", df_plot_prep["Product"].unique(),
                                                          key="line_plot_product")

            # Line Plot Chart
            with right_column:
                df_plot = line_plot_data_prep(df_plot_prep, industry_filter_selection, industry_segment_filter_selection,
                                              product_filter_selection, line_plot_group)

                plot = fig_line_plot(df_plot, colors, line_plot_group)
                st.altair_chart(plot, use_container_width=True)

        # UI Divider
        st.divider()

        # Industry Segment
        with st.container():
            st.subheader("Industry Segment Treemap")

            # Filters Selection
            with st.container():
                left_column, right_column = st.columns(2)

                with left_column:
                    year_filter_list = sorted(df_plot_prep["Fiscal_Year"].unique().tolist())
                    year_filter_list.insert(0, "All")

                    year_filter_selection = st.selectbox("Year Filter", year_filter_list,
                                                         key="treemap_year", index=0)

                with right_column:
                    product_filter_list = df_plot_prep["Product"].unique().tolist()
                    product_filter_list.insert(0, "All")

                    product_filter_selection = st.selectbox("Product Filter", product_filter_list,
                                                            key="treemap_product",)

            df_plot = industry_treemap_data_prep(df_plot_prep, year_filter_selection, product_filter_selection)

            plot = fig_industry_treemap(df_plot)
            st.plotly_chart(plot, use_container_width=True)