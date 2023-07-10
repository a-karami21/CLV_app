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
if "expected_lifetime" not in ss:
    ss.expected_lifetime = None

# App Title & Description
with st.container():
    app_title = config["app"]["app_title"]
    st.title(app_title)

    app_description = config["app"]["app_description"]
    st.markdown(app_description)

## Sidebar Options
# Data Upload
col1.header("1. Data")
with col1.expander("Dataset", expanded=True):
    # Uploader
    dataset_file = st.file_uploader('Upload the order intake data')

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

        with st.form("Attribute Selection"):
            df_date_column = st.selectbox("Select Date Attribute", ss.df_columns, key="Date")
            df_product_category_column = st.selectbox("Select Product Attribute", ss.df_columns, key="Product")
            df_customer_id_column = st.selectbox("Select Customer ID Attribute", ss.df_columns, key="Customer ID")
            df_customer_name_column = st.selectbox("Select Customer Name Attribute", ss.df_columns, key="Customer Name")
            df_industry_column = st.selectbox("Select Industry Attribute", ss.df_columns, key="Industry")
            df_monetary_column = st.selectbox("Select Monetary Attribute", ss.df_columns, key="Monetary Value")

            submitted = st.form_submit_button("Submit")

            # Snapshot the selected attributes in a dictionary
            ss.selected_columns_dict = {"Date": df_date_column,
                                     "Product": df_product_category_column,
                                     "Customer ID": df_customer_id_column,
                                     "Customer Name": df_customer_name_column,
                                     "Industry": df_industry_column,
                                     "Monetary Value": df_monetary_column}

            if submitted:
                for key, value in ss.selected_columns_dict.items():
                    if value == "Not Selected":
                        not_selected_list = []
                        st.write(f"Please select the {key} attributes")
                        ss.attribute_is_valid = False
                        break
                    else:
                        ss.attribute_is_valid = True

    # Prediction Inputs
    if ss.attribute_is_valid:

        # Rename the Columns for Standardization
        reversed_selected_columns_dict = {value: key for key, value in ss.selected_columns_dict.items()}
        ss.df0.rename(columns=reversed_selected_columns_dict, inplace=True)

        col1.header("3. Prediction Inputs")
        with col1.expander("User Inputs", expanded=True):

            # Get All Unique Values of Industry Columns
            industry_options = ss.df0["Industry"].unique()
            # Add "All" to Train All the Unique Values
            industry_options = np.insert(industry_options, 0, "All")
            # User Industry Selection for Training
            industry_selection = st.selectbox('Select industries to be trained on', industry_options, index=0)

            # Annual Discount Rate Selection
            annual_discount_rate = st.number_input("Input Annual Interest Rate (Default is 6%)",
                                                   min_value=0.01,
                                                   max_value=0.25,
                                                   value=0.06)

            # Expected Lifetime Selection
            expected_lifetime = st.slider("Select prediction lifetime (Months)",
                                          min_value=12,
                                          max_value=120,
                                          step=12)

            ss.expected_lifetime = expected_lifetime

        # # Dashboard Filters
        # col1.header("3. Dashboard Filters")
        # with col1.expander("Select Filter"):
        #     industry_type_cust = st.selectbox("Select Industry Type for Top 20 Customers", industry_options)
        #
        #     p_alive_slider = st.slider("Probability alive lower than X %", 10, 100, 80)
        #     ss.prob_alive_input = float(p_alive_slider / 100)

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
    with st.container():
        # Filter Dataframe to the Selected Industries
        if industry_selection == "All":
            pass
        else:
            ss.df0 = ss.df0["Industry"].isin([industry_selection])

        # Treat CustomerID as a categorical variable
        ss.df0["Customer ID"].astype(np.int64).astype(object)

        # Convert Date Column to Datetime Format
        ss.df0["Date"] = pd.to_datetime(ss.df0["Date"])

        # Convert Monetary Value Column to Numeric Format
        ss.df0["Monetary Value"] = pd.to_numeric(ss.df0["Monetary Value"])

        # Create Dictionary for Each Product Category & Its Dataframe
        ss.product_list = ss.df0["Product"].unique().tolist()
        ss.df_list = {}
        for product in ss.product_list:
            df = ss.df0[ss.df0["Product"].isin([product])]
            ss.df_list[product] = df

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

            with st.expander("Model Evaluation Result"):
                st.write("Training BG/NBD Model: does the model reflect the actual data closely enough?")
                tab_list = st.tabs(ss.product_list)
                for bu, tab in zip(ss.product_list, tab_list):
                    with tab:
                        # Predicted vs Actual (Train-Test) Graph Container
                        with st.container():
                            left_column, right_column = st.columns(2)

                            # Predicted vs Actual Chart By Number of Customer
                            with left_column:
                                fig1 = eva_viz1(ss.bgf_list[bu])
                                st.pyplot(fig1.figure)

                            # Predicted vs Actual By Holdout & Calibration Data Purchases
                            with right_column:
                                fig2 = plot_calibration_purchases_vs_holdout_purchases(ss.bgf_list[bu],
                                                                                       ss.df_ch_list[bu], n=9)
                                st.pyplot(fig2.figure)

                        st.divider()

                        # Model Performance Metrics
                        with st.container():
                            left_column, middle_column, right_column = st.columns(3)

                            # RMSE, MAE, Pearson Correlation, MAPE
                            with left_column:
                                with st.container():
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
                                with st.container():
                                    st.write("Prediction Difference of Customer Count")

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
                                with st.container():
                                    st.write("Prediction Difference of Calibration vs Holdout Purchases")

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

                    tab_list = st.tabs(ss.product_list)
                    for bu, tab in zip(ss.product_list, tab_list):
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
