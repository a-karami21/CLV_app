import numpy as np
import pandas as pd
from datetime import timedelta
import seaborn as sns
import streamlit as st
import io

from PIL import Image

from sklearn.metrics import mean_absolute_percentage_error

from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import \
    calibration_and_holdout_data, \
    summary_data_from_transaction_data

from lifetimes.plotting import \
    plot_period_transactions, \
    plot_calibration_purchases_vs_holdout_purchases

from scipy.stats import chi2_contingency
from scipy.stats import chi2

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
    st.title('Customer-Base Analysis App')
    st.write("This app is for analyzing customer-base purchase behavior and future value")

# Overview Section
    st.header("1. Overview")

# App Function & Guide
with st.expander("App Overview"):
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown("**App Functions**")
        st.markdown("* Determine high value customers")
        st.markdown("* Identify customer inactivity")
        st.markdown("* Analyze customer purchase behavior")
        st.markdown("* Predict customer lifetime value")
        st.markdown("* Predict industry growth")
        st.markdown("")

        st.markdown("**What is Customer Lifetime Value?**")
        st.markdown("Customer lifetime value is the **present value**"
                    " of the **future (net) cash flows** associated with a particular customer.")

        st.markdown("**What is RFM Metrics?**")
        st.markdown("RFM Metrics are metrics for measuring customer purchase characteristics.")
        st.markdown("* **Recency**: the age of the customer at the moment of his last purchase, "
                    "which is equal to the duration between a customer’s first purchase and their last purchase.")
        st.markdown("* **Frequency**: the number of periods in which the customer has made a repeat purchase.")
        st.markdown("* **T**: the age of the customer at the end of the period under study, "
                    "which is equal to the duration between a customer’s first purchase and the last day in the dataset.")
        st.markdown("* **Monetary**: the customer's average transaction value (order intake).")

    with right_column:
        st.markdown("**User Guide**")
        st.markdown("1. Upload the provided dataset* on the sidebar.")
        st.markdown("2. Select **[Business Unit]** and **[Industry Type]** to be trained.")
        st.markdown("3. Select desired **[Interest Rate]** and **[Lifetime (Months)]** you wish to predict.")
        st.markdown("4. Check the **Train Model** checkbox.")
        st.markdown("5. Open the **[Performance Score]** box to see model performance.")
        st.markdown("* **MAE** is the average error of the predicted model")
        st.markdown("Example: MAE value of 1 means the prediction is off by 1 transaction on average")
        st.markdown("* **RMSE** is the root average error of the predicted model similar to stdev")
        st.markdown("Example: High RMSE value means the prediction error has a high variance")
        st.markdown("* **Chi-Square test** is a test to identify if there are significant difference"
                    "between model and actual data.")
        st.markdown("* **Pearson Correlation** is the correlation between frequency and monetary.")
        st.markdown("The closer to zero the more appropriate the data for the model because"
                    " model assumes there are no dependency between frequency and monetary")
        st.markdown("* **MAPE** is the average percentage error of the prediction.")
        st.markdown("is used to calculate error between predicted and actual average transaction value")
        st.markdown("6. You can view or download the model **results & stats**.")
        st.markdown("7. Go to **[Visualization]** page in sidebar for visualization.")
        st.markdown("8. Input your desired **filters** in sidebar.")
        st.markdown("")

# App Workflow
with st.expander("App Workflow"):
    st.markdown("**Workflow**")
    image = Image.open('CLV_Flowchart.png')
    st.image(image, caption='Basic Workflow', width=800)

# Side Bar Inputs
with st.container():
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
    @st.experimental_memo
    def read_order_intake_csv():
        ss.df0 = pd.read_csv(orderintake_file, sep = ",", parse_dates=['order_intake_date'])
        return ss.df0

    ss.df0 = read_order_intake_csv()

if ss.df0 is not None:

    # Show Data Info
    with st.container():
        # treat CustomerID as a categorical variable
        ss.df0["ec_eu_customer"] = ss.df0["ec_eu_customer"].astype(np.int64).astype(object)

        # Filter to business unit input
        df1 = ss.df0[ss.df0['business_model_n'].isin(["Product"])]
        df2 = ss.df0[ss.df0['business_model_n'].isin(["Project (POC)"])]

        # Filter to industry type input
        ss.df_list = [df1, df2]

        def industry_filter(df):
            if industry_type_selection == "Energy":
                df = df[df['ec_eu_industry_type_n'].isin(["Energy"])]
            elif industry_type_selection == "Material":
                df = df[df['ec_eu_industry_type_n'].isin(["Material"])]
            elif industry_type_selection == "Life":
                df = df[df['ec_eu_industry_type_n'].isin(["Life"])]
            elif industry_type_selection == "All":
                pass
            return df

        ss.df_filtered_list = []
        for df in ss.df_list:
            ss.df_filtered_list.append(industry_filter(df))

    # Model Evaluation Section
    st.header("2. Model Evaluation")

    train_checkbox = st.checkbox("Train Model")

    # Training BG/NBD to Fitting on the full dataset
    with st.container():
        if train_checkbox:
            # BG/NBD Model

            # train/test split (calibration/holdout)
            def train_test_split(df):
                t_holdout = 365                                         # days to reserve for holdout period

                max_date = df["order_intake_date"].max()                     # end date of observations

                max_cal_date = max_date - timedelta(days=t_holdout)     # end date of chosen calibration period

                df_ch = calibration_and_holdout_data(
                        transactions = df,
                        customer_id_col = "ec_eu_customer",
                        datetime_col = "order_intake_date",
                        monetary_value_col = "order_intake_amount_lc",
                        calibration_period_end = max_cal_date,
                        observation_period_end = max_date,
                        freq = 'D',
                        )
                return df_ch

            def max_date_func(df):
                max_date = df["order_intake_date"].max()
                return max_date

            ss.df_ch_list = []
            for df in ss.df_filtered_list:
                ss.df_ch_list.append(train_test_split(df))

            max_date_list = []
            for df in ss.df_filtered_list:
                max_date_list.append(max_date_func(df))

            # determine recency, frequency, T, monetary value for each customer
            def determine_df_rft(df, max_date):
                df_rft = summary_data_from_transaction_data(
                    transactions=df,
                    customer_id_col="ec_eu_customer",
                    datetime_col="order_intake_date",
                    monetary_value_col="order_intake_amount_lc",
                    observation_period_end=max_date,
                    freq="D")

                return df_rft

            ss.df_rft_list = []
            for df, max_date in zip(ss.df_list, max_date_list):
                ss.df_rft_list.append(determine_df_rft(df, max_date))

            # Fit the BG/NBD Model
            @st.cache_resource
            def bgf_fit(df_ch):
                bgf = BetaGeoFitter(penalizer_coef=1e-06)
                bgf.fit(
                    frequency=df_ch["frequency_cal"],
                    recency=df_ch["recency_cal"],
                    T=df_ch["T_cal"])

                return bgf

            ss.bgf_list = []
            for df_ch in ss.df_ch_list:
                ss.bgf_list.append(bgf_fit(df_ch))

            def bgf_evaluation_prep(bgf, df_ch):
                # get predicted frequency during holdout period
                frequency_holdout_predicted = bgf.predict(df_ch['duration_holdout'],
                                                          df_ch['frequency_cal'],
                                                          df_ch['recency_cal'],
                                                          df_ch['T_cal'])

                # get actual frequency during holdout period
                frequency_holdout_actual = df_ch['frequency_holdout']

                return frequency_holdout_predicted, frequency_holdout_actual


            bgf_evaluation_list = []
            for bgf, df_ch in zip(ss.bgf_list, ss.df_ch_list):
                bgf_evaluation_list.append(bgf_evaluation_prep(bgf, df_ch))

            holdout_predicted_list = [bgf_evaluation_list[0][0], bgf_evaluation_list[1][0]]
            holdout_actual_list = [bgf_evaluation_list[0][1], bgf_evaluation_list[1][1]]

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

            # BG/NBD model fitting
            @st.cache_resource
            def bgf_full_fit(df_rft):
                bgf = BetaGeoFitter(penalizer_coef=1e-06)
                bgf.fit(
                    frequency=df_rft["frequency"],
                    recency=df_rft["recency"],
                    T=df_rft["T"])

                return bgf

            ss.bgf_full_list = []
            for df_rft in ss.df_rft_list:
                ss.bgf_full_list.append(bgf_full_fit(df_rft))

            def predict_purch(df, bgf, t):
                df["predict_purch_" + str(t)] = \
                    bgf.predict(
                        t,
                        df["frequency"],
                        df["recency"],
                        df["T"])

            # call helper function: predict each customer's purchases over multiple time periods
            t_FC = [90, 180, 270, 360]

            for t in t_FC:
                for df_rft, bgf in zip(ss.df_rft_list, ss.bgf_list):
                    predict_purch(df_rft, bgf, t)

            # probability that a customer is alive for each customer in dataframe
            def calculate_prob_alive(bgf, df_rft):
                prob_alive = bgf.conditional_probability_alive(
                    frequency=df_rft["frequency"],
                    recency=df_rft["recency"],
                    T=df_rft["T"])

                return prob_alive

            for bgf, df_rft in zip(ss.bgf_full_list, ss.df_rft_list):
                df_rft["prob_alive"] = calculate_prob_alive(bgf, df_rft)

            def gg_prep(df_rft):
                # select customers with monetary value > 0
                df_rftv = df_rft[df_rft["monetary_value"] > 0]
                return df_rftv

            def gg_corr(df_rftv):
                # Gamma-Gamma model requires a Pearson correlation close to 0
                # between purchase frequency and monetary value
                corr_matrix = df_rftv[["monetary_value", "frequency"]].corr()
                corr = corr_matrix.iloc[1, 0]
                return corr

            ss.df_rftv_list = []
            for df_rft in ss.df_rft_list:
                ss.df_rftv_list.append(gg_prep(df_rft))

            corr_list = []
            for df_rftv in ss.df_rftv_list:
                corr_list.append(gg_corr(df_rftv))

            # fitting the Gamma-Gamma model
            @st.cache_resource
            def gg_fit(df_rftv):
                ggf = GammaGammaFitter(penalizer_coef=0.0001)
                ggf.fit(
                    frequency=df_rftv["frequency"],
                    monetary_value=df_rftv["monetary_value"])

                return ggf

            ss.ggf_list = []
            for df_rftv in ss.df_rftv_list:
                ss.ggf_list.append(gg_fit(df_rftv))

            # estimate the average transaction value of each customer, based on frequency and monetary value
            def gg_avg(df_rftv, ggf):
                exp_avg_rev = ggf.conditional_expected_average_profit(
                    df_rftv["frequency"],
                    df_rftv["monetary_value"]
                )

                df_rftv["exp_avg_rev"] = exp_avg_rev
                df_rftv["avg_rev"] = df_rftv["monetary_value"]
                df_rftv["error_rev"] = df_rftv["exp_avg_rev"] - df_rftv["avg_rev"]

                return df_rftv

            def gg_evaluation(df_rftv):
                mape = mean_absolute_percentage_error(df_rftv["exp_avg_rev"], df_rftv["monetary_value"])
                return mape

            for df_rftv, ggf in zip(ss.df_rftv_list, ss.ggf_list):
                ss.df_rftv_list.append(gg_avg(df_rftv, ggf))

            for x in range(2):
                ss.df_rftv_list.pop(0)

            mape_list = []
            for df_rftv in ss.df_rftv_list:
                mape_list.append(gg_evaluation(df_rftv))

            # compute customer lifetime value
            @st.cache_resource
            def compute_clv(df_rftv, _ggf, _bgf):
                DISCOUNT_a = annual_discount_rate  # annual discount rate
                LIFE = ss.expected_lifetime  # lifetime expected for the customers in months

                discount_m = (1 + DISCOUNT_a) ** (1 / 12) - 1  # monthly discount rate

                clv = ggf.customer_lifetime_value(
                    transaction_prediction_model= bgf,
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

            for df_rftv, ggf, bgf in zip(ss.df_rftv_list, ss.ggf_list, ss.bgf_full_list):
                ss.df_rftv_list.append(compute_clv(df_rftv, ggf, bgf))

            for x in range(2):
                ss.df_rftv_list.pop(0)

            # Predicted vs Actual (Train-Test) Graph Container
            with st.container():
                st.write("Training BG/NBD Model: does the model reflect the actual data closely enough?")
                left_column, right_column = st.columns(2)
                with left_column:
                    def eva_viz1(bgf):
                        import matplotlib.pyplot as plt

                        n = bgf.data.shape[0]
                        simulated_data = bgf.generate_new_data(size=n)

                        model_counts = pd.DataFrame(bgf.data["frequency"].value_counts().sort_index().iloc[:9])
                        simulated_counts = pd.DataFrame(
                            simulated_data["frequency"].value_counts().sort_index().iloc[:9])
                        combined_counts = model_counts.merge(simulated_counts, how="outer", left_index=True,
                                                             right_index=True).fillna(0)
                        combined_counts.columns = ["Actual", "Model"]

                        ax = combined_counts.plot(kind="bar")

                        ax.bar_label(ax.containers[0], label_type='edge')
                        ax.bar_label(ax.containers[1], label_type='edge')

                        plt.legend()
                        plt.title("Frequency of Repeat Transactions")
                        plt.ylabel("Customers")
                        plt.xlabel("Number of Calibration Period Transactions")

                        return ax

                    tab1, tab2 = st.tabs(["Product", "Project"])
                    with tab1:
                        fig1 = eva_viz1(ss.bgf_list[0])
                        st.pyplot(fig1.figure)

                    with tab2:
                        fig1 = eva_viz1(ss.bgf_list[1])
                        st.pyplot(fig1.figure)

                with right_column:
                    tab1, tab2 = st.tabs(["Product", "Project"])
                    with tab1:
                        fig2 = plot_calibration_purchases_vs_holdout_purchases(ss.bgf_list[0], ss.df_ch_list[0], n=9)
                        st.pyplot(fig2.figure)
                    with tab2:
                        fig2 = plot_calibration_purchases_vs_holdout_purchases(ss.bgf_list[1], ss.df_ch_list[1], n=9)
                        st.pyplot(fig2.figure)

            with st.container():
                left_column2, middle_column2, right_column2 = st.columns(3)
                with left_column2:
                    with st.expander("Performance Score"):
                        tab1, tab2 = st.tabs(["Product", "Project"])
                        with tab1:
                            st.write("BG/NBD Model Performance:")
                            st.markdown("* MAE: {0}".format(score_model(holdout_actual_list[0],
                                                                        holdout_predicted_list[0],
                                                                        'mae')))
                            st.markdown("* RMSE: {0}".format(score_model(holdout_actual_list[0],
                                                                         holdout_predicted_list[0],
                                                                         'rmse')))
                            st.write("Gamma-Gamma Model Performance:")
                            st.markdown("* Pearson correlation: %.3f" % corr_list[0])
                            st.markdown("* MAPE of predicted revenues: " + f'{mape_list[0]:.2f}')
                        with tab2:
                            st.write("BG/NBD Model Performance:")
                            st.markdown("* MAE: {0}".format(score_model(holdout_actual_list[1],
                                                                        holdout_predicted_list[1],
                                                                        'mae')))
                            st.markdown("* RMSE: {0}".format(score_model(holdout_actual_list[1],
                                                                         holdout_predicted_list[1],
                                                                         'rmse')))
                            st.write("Gamma-Gamma Model Performance:")
                            st.markdown("* Pearson correlation: %.3f" % corr_list[1])
                            st.markdown("* MAPE of predicted revenues: " + f'{mape_list[1]:.2f}')

                with middle_column2:
                    with st.expander("Prediction Difference of Customer Count"):
                        tab1, tab2 = st.tabs(["Product", "Project"])
                        with tab1:
                            import matplotlib.pyplot as plt

                            n = ss.bgf_list[0].data.shape[0]
                            simulated_data2 = ss.bgf_list[0].generate_new_data(size=n)

                            model_counts2 = pd.DataFrame(ss.bgf_list[0].data["frequency"].value_counts().sort_index().iloc[:])
                            simulated_counts2 = pd.DataFrame(simulated_data2["frequency"].value_counts().sort_index().iloc[:])
                            combined_counts2 = model_counts2.merge(simulated_counts2, how="outer", left_index=True,
                                                                     right_index=True).fillna(0)
                            combined_counts2.columns = ["Actual", "Model"]

                            df_viz1 = combined_counts2
                            df_viz1['Difference'] = combined_counts2['Actual'] - combined_counts2['Model']

                            df_viz1.index.name = "Transaction"

                            df_viz1_9 = df_viz1[8:]
                            df_viz1_9 = df_viz1_9.mean(axis=0)
                            df_viz1_9.name = "9+"

                            combined_counts_final = df_viz1[:8].append(df_viz1_9)

                            st._legacy_dataframe(combined_counts_final.style.format("{:,.0f}"))

                            chi, pval, dof, exp = chi2_contingency(combined_counts_final.iloc[:, :2])

                            significance = 0.001
                            p = 1 - significance
                            critical_value = chi2.ppf(p, dof)

                            st.markdown('p-value is {:.5f}'.format(pval))
                            st.markdown('chi = %.6f, critical value = %.6f' % (chi, critical_value))
                            if chi > critical_value:
                                st.markdown("""At %.3f level of significance, we reject the null hypotheses and accept H1. 
                              There is significant difference between actual and model.""" % (significance))
                            else:
                                st.markdown("""At %.3f level of significance, we accept the null hypotheses. 
                              There is no significant difference between actual and model.""" % (significance))

                        with tab2:
                            import matplotlib.pyplot as plt

                            n = ss.bgf_list[1].data.shape[1]
                            simulated_data2 = ss.bgf_list[1].generate_new_data(size=n)

                            model_counts2 = pd.DataFrame(
                                ss.bgf_list[1].data["frequency"].value_counts().sort_index().iloc[:])
                            simulated_counts2 = pd.DataFrame(
                                simulated_data2["frequency"].value_counts().sort_index().iloc[:])
                            combined_counts2 = model_counts2.merge(simulated_counts2, how="outer", left_index=True,
                                                                   right_index=True).fillna(0)
                            combined_counts2.columns = ["Actual", "Model"]

                            df_viz1 = combined_counts2
                            df_viz1['Difference'] = combined_counts2['Actual'] - combined_counts2['Model']

                            df_viz1.index.name = "Transaction"

                            df_viz1_9 = df_viz1[8:]
                            df_viz1_9 = df_viz1_9.mean(axis=0)
                            df_viz1_9.name = "9+"

                            combined_counts_final = df_viz1[:8].append(df_viz1_9)

                            st._legacy_dataframe(combined_counts_final.style.format("{:,.0f}"))

                            chi, pval, dof, exp = chi2_contingency(combined_counts_final.iloc[:, :2])

                            significance = 0.001
                            p = 1 - significance
                            critical_value = chi2.ppf(p, dof)

                            st.markdown('p-value is {:.5f}'.format(pval))
                            st.markdown('chi = %.6f, critical value = %.6f' % (chi, critical_value))
                            if chi > critical_value:
                                st.markdown("""At %.3f level of significance, we reject the null hypotheses and accept H1. 
                              There is significant difference between actual and model.""" % (significance))
                            else:
                                st.markdown("""At %.3f level of significance, we accept the null hypotheses. 
                              There is no significant difference between actual and model.""" % (significance))

                with right_column2:
                    with st.expander("Prediction Difference of Calibration vs Holdout Purchases"):
                        tab1, tab2 = st.tabs(["Product", "Project"])
                        with tab1:
                            from matplotlib import pyplot as plt

                            kind = "frequency_cal"

                            x_labels = {
                                "frequency_cal": "Purchases in calibration period",
                                "recency_cal": "Age of customer at last purchase",
                                "T_cal": "Age of customer at the end of calibration period",
                                "time_since_last_purchase": "Time since user made last purchase",
                            }

                            summary = ss.df_ch_list[0].copy()
                            duration_holdout = summary.iloc[0]["duration_holdout"]

                            summary["model_predictions"] = ss.bgf_list[0].conditional_expected_number_of_purchases_up_to_time(
                                duration_holdout, summary["frequency_cal"], summary["recency_cal"], summary["T_cal"])

                            purch_diff = summary.groupby(kind)[["frequency_holdout", "model_predictions"]].mean().iloc[:]

                            purch_diff['Difference'] = purch_diff['model_predictions'] - purch_diff['frequency_holdout']

                            df_viz2 = purch_diff

                            df_viz2.index.name = "Transaction"

                            df_viz2_9 = df_viz2[8:]
                            df_viz2_9 = df_viz2_9.mean(axis=0)
                            df_viz2_9.name = "9+"

                            purch_diff_final = df_viz2[:8].append(df_viz2_9)

                            st._legacy_dataframe(purch_diff_final.style.format("{:,.2f}"))

                            chi, pval, dof, exp = chi2_contingency(purch_diff_final.iloc[:, :2])

                            st.markdown('p-value is {:.5f}'.format(pval))

                            significance = 0.001
                            p = 1 - significance
                            critical_value = chi2.ppf(p, dof)

                            st.markdown('chi = %.6f, critical value = %.6f' % (chi, critical_value))

                            if chi > critical_value:
                                st.markdown("""At %.3f level of significance, we reject the null hypotheses and accept H1. 
                            There is significant difference between actual and model.""" % (significance))
                            else:
                                st.markdown("""At %.3f level of significance, we accept the null hypotheses. 
                            There is no significant difference between actual and model.""" % (significance))

                        with tab2:
                            from matplotlib import pyplot as plt

                            kind = "frequency_cal"

                            x_labels = {
                                "frequency_cal": "Purchases in calibration period",
                                "recency_cal": "Age of customer at last purchase",
                                "T_cal": "Age of customer at the end of calibration period",
                                "time_since_last_purchase": "Time since user made last purchase",
                            }

                            summary = ss.df_ch_list[1].copy()
                            duration_holdout = summary.iloc[1]["duration_holdout"]

                            summary["model_predictions"] = ss.bgf_list[
                                0].conditional_expected_number_of_purchases_up_to_time(
                                duration_holdout, summary["frequency_cal"], summary["recency_cal"], summary["T_cal"])

                            purch_diff = summary.groupby(kind)[["frequency_holdout", "model_predictions"]].mean().iloc[
                                         :]

                            purch_diff['Difference'] = purch_diff['model_predictions'] - purch_diff['frequency_holdout']

                            df_viz2 = purch_diff

                            df_viz2.index.name = "Transaction"

                            df_viz2_9 = df_viz2[8:]
                            df_viz2_9 = df_viz2_9.mean(axis=0)
                            df_viz2_9.name = "9+"

                            purch_diff_final = df_viz2[:8].append(df_viz2_9)

                            st._legacy_dataframe(purch_diff_final.style.format("{:,.2f}"))

                            chi, pval, dof, exp = chi2_contingency(purch_diff_final.iloc[:, :2])

                            st.markdown('p-value is {:.5f}'.format(pval))

                            significance = 0.001
                            p = 1 - significance
                            critical_value = chi2.ppf(p, dof)

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

                    def export_table(df_rftv, df):
                        df_final = df_rftv

                        df_final = df_final.reset_index().merge(df[["ec_eu_customer",
                                                                        "ec_eu_customer_n",
                                                                        "ec_eu_industry_type_n",
                                                                        "ec_eu_industry_sub_segment_n",
                                                                        "ec_eu_industry_segment_n"]],
                                                                how="left").set_index('ec_eu_customer')

                        order = ['ec_eu_customer_n', 'ec_eu_industry_type_n', 'ec_eu_industry_sub_segment_n',
                                 'ec_eu_industry_segment_n','CLV', 'frequency', 'recency', 'T', 'monetary_value',
                                 'predict_purch_90', 'predict_purch_180', 'predict_purch_270', 'predict_purch_360',
                                 'prob_alive', 'exp_avg_rev', 'avg_rev', 'error_rev']

                        df_final = df_final[order]
                        df_final = df_final.drop_duplicates()
                        df_final = df_final.rename_axis('ec_eu_customer').reset_index()

                        return df_final

                    ss.df_viz_list = []
                    for df_final, df in zip(ss.df_rftv_list, ss.df_filtered_list):
                        ss.df_viz_list.append(export_table(df_final, df))

                    tab1, tab2 = st.tabs(["Product", "Project"])
                    with tab1:

                        df_viz = ss.df_viz_list[0]

                        st.dataframe(df_viz.style.format({
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

                        @st.cache_resource
                        def convert_df(df):
                            return df.to_csv().encode('utf-8')

                        df_csv = convert_df(df_viz)

                        st.download_button("Press to Download",
                                           df_csv,
                                           "CLV Table.csv",
                                           "text/csv",
                                           key='download-csv-product'
                                           )

                        # Display full table from the result
                        st.write("Result Stats")

                        st.dataframe(df_viz.describe().style.format("{:,.2f}"))

                    with tab2:
                        df_viz = ss.df_viz_list[1]

                        st.dataframe(df_viz.style.format({
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

                        @st.cache_resource
                        def convert_df(df):
                            return df.to_csv().encode('utf-8')

                        df_csv = convert_df(df_viz)

                        st.download_button("Press to Download",
                                           df_csv,
                                           "CLV Table.csv",
                                           "text/csv",
                                           key='download-csv-project'
                                           )

                        # Display full table from the result
                        st.write("Result Stats")

                        st.dataframe(df_viz.describe().style.format("{:,.2f}"))
