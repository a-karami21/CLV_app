import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import seaborn as sns
import streamlit as st
import io

from PIL import Image
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource

from sklearn.metrics import mean_absolute_percentage_error

from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import \
    calibration_and_holdout_data, \
    summary_data_from_transaction_data

from lifetimes.plotting import \
    plot_period_transactions, \
    plot_calibration_purchases_vs_holdout_purchases

from bokeh.models import FactorRange


sns.set(rc={'image.cmap': 'coolwarm'})

# Page Config
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

col1 = st.sidebar
col2, col3 = st.columns((2,1))

# Initialization
ss = st.session_state

# initialize session state variable
if "df0" not in ss:
    ss.df0 = None
if "df1" not in ss:
    ss.df1 = None
if "df_ch" not in ss:
    ss.df_ch = None
if "df_rft" not in ss:
    ss.df_rft = None
if "bgf" not in ss:
    ss.bgf = None
if "df_rftv" not in ss:
    ss.df_rftv = None
if "df_viz" not in ss:
    ss.df_viz = None

if "customer_id_input" not in ss:
    ss.customer_id_input = None
if "customer_name_input" not in ss:
    ss.customer_name_input = None
if "prob_alive_input" not in ss:
    ss.prob_alive_input = None
if "expected_lifetime" not in ss:
    ss.expected_lifetime = None

with st.container():
    # Main Panel Title
    st.title('Customer Lifetime Value Web App')

    st.write("This app is for predicting and monitoring customer lifetime value")

with st.container():
    with col1.container():
        # Sidebar Title
        col1.header("1. Data")
        with col1.expander("Dataset", expanded=True):
            # Uploader
            orderintake_file = st.file_uploader('Upload the order intake data')

            # Sidebar Options
            business_unit_selection = st.selectbox('Select business unit', ('Product', 'Project', 'Both'), index=0)

            # Sidebar Options
            industry_type_selection = st.selectbox('Select industry type', ('Energy', 'Material', 'Life', 'All'), index=3)

        col1.header("2. Prediction Inputs")
        with col1.expander("User Inputs"):
            # Predict Purchase Range Number
            annual_discount_rate = st.number_input("Input Annual Interest Rate (Default is 6%)", 0.01, 0.25, value=0.06)
            expected_lifetime = st.slider("Select prediction lifetime (Months)", 12, 120, step=12)

            ss.expected_lifetime = expected_lifetime

# Data Loading

if orderintake_file is not None and ss.df0 is None:
    @st.experimental_memo
    def read_order_intake_csv():
        dtypes = {'year_month': 'str'}
        df0 = pd.read_csv(orderintake_file, dtype=dtypes, parse_dates=['year_month'])
        df0 = df0.rename(columns={'eu_industry_segment_n': 'eu_industry_sub_segment_n',
                                  'eu_industry_sub_segment_n': 'eu_industry_segment_n'})
        df0['order_intake_amount_lc'] = df0['order_intake_amount_lc'].replace({'\$': '', ',': ''}, regex=True).astype(
            float)

        return df0

    ss.df0 = read_order_intake_csv()

if ss.df0 is not None:
    # Overview Section
    st.header("1. Overview")

    # App Function & Guide
    with st.expander("App Overview"):
        left_column, right_column = st.columns(2)
        with left_column:
            st.markdown("**App Functions**")
            st.markdown("* Determine high value customers")
            st.markdown("* Detect customer inactivity")
            st.markdown("* Analyze customer purchase behavior")
            st.markdown("* Predict customer lifetime value")
            st.markdown("* Predict industry growth")
            st.markdown("")

            st.markdown("**User Guide**")
            st.markdown("1. Upload the provided dataset on the sidebar.")
            st.markdown("2. Select **[Business Unit]** and **[Industry Type]** to be trained.")
            st.markdown("3. Select desired **[Interest Rate]** and **[Lifetime (Months)]** you wish to predict.")
            st.markdown("4. Check the **Train Model** checkbox.")
            st.markdown("5. Open the **[Performance Score]** box.")
            st.markdown("  * BG/NBD Model performance is **good** if RMSE below **1.5**, **great** if below **1.0**.")
            st.markdown("  * Gamma-Gamma Model performance is **good** if pearson correlation below **0.5** and MAPE **close to zero**.")
            st.markdown("6. You can view the model **results & stats**.")
            st.markdown("7. Go to **[Visualization]** page in sidebar.")
            st.markdown("8. Input your desired **filters** in sidebar.")
            st.markdown("")

            st.markdown("**What is Customer Lifetime Value?**")
            st.markdown("Customer lifetime value is the **present value**"
                        " of the **future (net) cash flows** associated with a particular customer")

        with right_column:
            st.markdown("**Workflow**")

            image = Image.open('CLV_Flowchart.png')
            st.image(image, caption='Basic Workflow')

            st.markdown("**What is RFM & T?**")

            st.markdown("* **Recency**: the age of the customer at the moment of his last purchase, "
                        "which is equal to the duration between a customer’s first purchase and their last purchase.")
            st.markdown("* **Frequency**: the number of periods in which the customer has made a repeat purchase.")
            st.markdown("* **Monetary**: the average of purchase transaction value.")
            st.markdown("* **T**: the age of the customer at the end of the period under study, "
                        "which is equal to the duration between a customer’s first purchase and the last day in the dataset.")

    # Show Data Info
    with st.container():
        # Delete columns that are not useful
        try:
            d1 = ss.df0.drop(["fu", "fu_n", "sales_office", "sales_office_n", "sales_person", "business_model"], axis=1,
                          inplace=True)
        except:
            pass

        # Remove transactions with 0 OI amount
        df1 = ss.df0[ss.df0["order_intake_amount_lc"] > 0]

        # treat CustomerID as a categorical variable
        df1["ec_eu_customer"] = df1["ec_eu_customer"].astype(np.int64).astype(object)

        # Filter to business unit input
        if business_unit_selection == "Product":
            df1 = df1[df1['business_model_n'].isin(["Product"])]
        elif business_unit_selection == "Project":
            df1 = df1[df1['business_model_n'].isin(["Project (POC)"])]
        elif business_unit_selection == "Both":
            df1 = df1[df1.business_model_n != "Service"]

        # Filter to industry type input
        if industry_type_selection == "Energy":
            df1 = df1[df1['eu_industry_type_n'].isin(["Energy"])]
        elif industry_type_selection == "Material":
            df1 = df1[df1['eu_industry_type_n'].isin(["Material"])]
        elif industry_type_selection == "Life":
            df1 = df1[df1['eu_industry_type_n'].isin(["Life"])]
        elif industry_type_selection == "All":
            pass

        ss.df1 = df1

    # Model Evaluation Section
    st.header("2. Model Evaluation")

    train_checkbox = st.checkbox("Train Model")

    # Training BG/NBD to Fitting on the full dataset
    with st.container():
        if train_checkbox:
            # BG/NBD Model

            # train/test split (calibration/holdout)
            def train_test_split(df1):
                t_holdout = 365                                         # days to reserve for holdout period

                max_date = df1["year_month"].max()                     # end date of observations

                max_cal_date = max_date - timedelta(days=t_holdout)     # end date of chosen calibration period

                df_ch = calibration_and_holdout_data(
                        transactions = df1,
                        customer_id_col = "ec_eu_customer",
                        datetime_col = "year_month",
                        monetary_value_col = "order_intake_amount_lc",
                        calibration_period_end = max_cal_date,
                        observation_period_end = max_date,
                        freq = 'M',
                        )

                ss.df_ch = df_ch

                return df_ch, max_date

            df_ch, max_date = train_test_split(ss.df1)

            # Fit the BG/NBD Model
            @st.cache
            def bgf_fit(df_ch):
                bgf = BetaGeoFitter(penalizer_coef=1e-06)
                bgf.fit(
                    frequency=df_ch["frequency_cal"],
                    recency=df_ch["recency_cal"],
                    T=df_ch["T_cal"],
                    weights=None,
                    verbose=True,
                    tol=1e-06)

                return bgf

            bgf = bgf_fit(df_ch)

            # determine recency, frequency, T, monetary value for each customer
            def determine_df_rft(df1, max_date):
                df_rft = summary_data_from_transaction_data(
                    transactions=df1,
                    customer_id_col="ec_eu_customer",
                    datetime_col="year_month",
                    monetary_value_col="order_intake_amount_lc",
                    observation_period_end=max_date,
                    freq="D")

                return df_rft

            # Return df_rft to session state
            ss.df_rft = determine_df_rft(df1, max_date)

            # BG/NBD model fitting
            @st.cache
            def bgf_full_fit():
                bgf = BetaGeoFitter(penalizer_coef=1e-06)
                bgf.fit(
                    frequency=ss.df_rft["frequency"],
                    recency=ss.df_rft["recency"],
                    T=ss.df_rft["T"],
                    weights=None,
                    verbose=True,
                    tol=1e-06)

                return bgf

            # Return bgf model to session state
            ss.bgf = bgf_full_fit()

            def predict_purch(df, t):
                df["predict_purch_" + str(t)] = \
                    ss.bgf.predict(
                        t,
                        df["frequency"],
                        df["recency"],
                        df["T"])

            # call helper function: predict each customer's purchases over multiple time periods
            t_FC = [90, 180, 270, 360]
            _ = [predict_purch(ss.df_rft, t) for t in t_FC]
            #st.write("predicted number of purchases for each customer over next t days:")
            #st.dataframe(ss.df_rft.style.format("{:,.1f}"))

            # probability that a customer is alive for each customer in dataframe
            def calculate_prob_alive():
                prob_alive = ss.bgf.conditional_probability_alive(
                    frequency=ss.df_rft["frequency"],
                    recency=ss.df_rft["recency"],
                    T=ss.df_rft["T"])

                return prob_alive

            ss.df_rft["prob_alive"] = calculate_prob_alive()

            def bgf_evaluation_prep(bgf, df_ch):
                # get predicted frequency during holdout period
                frequency_holdout_predicted = bgf.predict(df_ch['duration_holdout'],
                                                          df_ch['frequency_cal'],
                                                          df_ch['recency_cal'],
                                                          df_ch['T_cal'])

                # get actual frequency during holdout period
                frequency_holdout_actual = df_ch['frequency_holdout']

                return frequency_holdout_predicted, frequency_holdout_actual

            frequency_holdout_predicted, frequency_holdout_actual = bgf_evaluation_prep(bgf, df_ch)

            # RMSE/MSE score
            def score_model(actuals, predicted, metric='rmse'):
                # make sure metric name is lower case
                metric = metric.lower()
                # Mean Squared Error and Root Mean Squared Error
                if metric == 'mse' or metric == 'rmse':
                    val = np.sum(np.square(actuals - predicted)) / actuals.shape[0]
                    if metric == 'rmse':
                        val = np.sqrt(val)
                # Mean Absolute Error
                elif metric == 'mae':
                    val = np.sum(np.abs(actuals - predicted)) / actuals.shape[0]
                else:
                    val = None
                return val

            def gg_prep():
                # select customers with monetary value > 0
                df_rftv = ss.df_rft[ss.df_rft["monetary_value"] > 1]

                # Gamma-Gamma model requires a Pearson correlation close to 0
                # between purchase frequency and monetary value
                corr_matrix = df_rftv[["monetary_value", "frequency"]].corr()
                corr = corr_matrix.iloc[1, 0]

                return df_rftv, corr

            df_rftv, corr = gg_prep()

            # fitting the Gamma-Gamma model
            @st.cache
            def gg_fit():
                ggf = GammaGammaFitter(penalizer_coef=0.001)
                ggf.fit(
                    frequency=df_rftv["frequency"],
                    monetary_value=df_rftv["monetary_value"])

                return ggf

            ggf = gg_fit()

            # estimate the average transaction value of each customer, based on frequency and monetary value
            def gg_evaluation(df_rftv):
                exp_avg_rev = ggf.conditional_expected_average_profit(
                    df_rftv["frequency"],
                    df_rftv["monetary_value"]
                )

                df_rftv["exp_avg_rev"] = exp_avg_rev
                df_rftv["avg_rev"] = df_rftv["monetary_value"]
                df_rftv["error_rev"] = df_rftv["exp_avg_rev"] - df_rftv["avg_rev"]

                mape = mean_absolute_percentage_error(exp_avg_rev, df_rftv["monetary_value"])

                return df_rftv, mape

            df_rftv, mape = gg_evaluation(df_rftv)

            # compute customer lifetime value
            @st.cache
            def compute_clv(df_rftv):
                DISCOUNT_a = annual_discount_rate  # annual discount rate
                LIFE = ss.expected_lifetime  # lifetime expected for the customers in months

                discount_m = (1 + DISCOUNT_a) ** (1 / 12) - 1  # monthly discount rate

                clv = ggf.customer_lifetime_value(
                    transaction_prediction_model=ss.bgf,
                    frequency=df_rftv["frequency"],
                    recency=df_rftv["recency"],
                    T=df_rftv["T"],
                    monetary_value=df_rftv["monetary_value"],
                    time=LIFE,
                    freq="D",
                    discount_rate=discount_m)

                df_rftv.insert(0, "CLV", clv)  # expected customer lifetime values
                df_rftv = df_rftv.sort_values(by="CLV", ascending=False)

                return df_rftv

            ss.df_rftv = compute_clv(df_rftv)

            # Predicted vs Actual (Train-Test) Graph Container
            with st.container():
                st.write("Training BG/NBD Model: does the model reflect the actual data closely enough?")
                left_column, right_column = st.columns(2)
                with left_column:
                    fig1 = plot_period_transactions(bgf)
                    st.pyplot(fig1.figure)
                with right_column:
                    fig2 = plot_calibration_purchases_vs_holdout_purchases(bgf, df_ch)
                    st.pyplot(fig2.figure)

            # Predicted vs Actual Full Dataset Graph Container
            with st.container():
                st.write("Predicted vs Actual on the Full Dataset")
                fig3 = plot_period_transactions(
                    model=ss.bgf,
                    max_frequency=10,
                    figsize=(20,6))
                st.pyplot(fig3.figure)

            with st.expander("Performance Score"):
                st.write("BG/NBD Model Performance:")
                st.markdown("* MAE: {0}".format(score_model(frequency_holdout_actual, frequency_holdout_predicted, 'mae')))
                st.markdown("* RMSE: {0}".format(score_model(frequency_holdout_actual, frequency_holdout_predicted, 'rmse')))
                st.write("Gamma-Gamma Model Performance:")
                st.markdown("* Pearson correlation: %.3f" % corr)
                st.markdown("* MAPE of predicted revenues: " + f'{mape:.2f}')

        with st.expander("Show Result"):
            if ss.df_rft is not None:
                # Display full table from the result
                st.write("Full Table")

                def export_table(df_rftv):
                    df_final = df_rftv

                    df_final = df_final.reset_index().merge(ss.df1[["ec_eu_customer",
                                                                    "ec_eu_customer_n",
                                                                    "eu_industry_type_n",
                                                                    "eu_industry_segment_n"]],
                                                            how="left").set_index('ec_eu_customer')

                    order = ['ec_eu_customer_n', 'eu_industry_type_n', 'eu_industry_segment_n',
                             'CLV', 'frequency', 'recency', 'T', 'monetary_value', 'predict_purch_90',
                             'predict_purch_180', 'predict_purch_270', 'predict_purch_360',
                             'prob_alive', 'exp_avg_rev', 'avg_rev', 'error_rev']

                    df_final = df_final[order]
                    df_final = df_final.drop_duplicates()
                    df_final = df_final.rename_axis('ec_eu_customer').reset_index()

                    st.dataframe(df_final.style.format({
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

                    return df_final

                ss.df_viz = export_table(ss.df_rftv)

                @st.cache
                def convert_df(df):
                    return df.to_csv().encode('utf-8')

                df_csv = convert_df(ss.df_viz)

                st.download_button("Press to Download",
                                   df_csv,
                                   "CLV Table.csv",
                                   "text/csv",
                                   key='download-csv'
                                   )

                # Display full table from the result
                st.write("Result Stats")

                st.dataframe(ss.df_rftv.describe().style.format("{:,.2f}"))