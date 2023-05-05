import seaborn as sns
import streamlit as st

from PIL import Image

from lifetimes.plotting import \
    plot_period_transactions, \
    plot_calibration_purchases_vs_holdout_purchases

from scipy.stats import chi2_contingency
from scipy.stats import chi2

# Page Config
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

sns.set(rc={'image.cmap': 'coolwarm'})

from src.lib.preparations import *
from src.lib.visualizations import *

col1 = st.sidebar
col2, col3 = st.columns((2,1))

# Initialization
ss = st.session_state

# initialize session state variable
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
    image = Image.open('./references/CLV_Flowchart.png')
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

    train_checkbox = st.checkbox("Train Model")

    # Training BG/NBD to Fitting on the full dataset
    with st.container():
        if train_checkbox:
            # BG/NBD Model

            ss.df_ch_list = {}
            for bu, df in ss.df_list.items():
                df_ch = train_test_split(df)
                ss.df_ch_list[bu] = df_ch

            ss.max_date_list = {}
            for bu, df in ss.df_list.items():
                max_date = max_date_func(df)
                ss.max_date_list[bu] = max_date

            ss.df_rft_list = {}
            for (bu, df), (bu, max_date) in zip(ss.df_list.items(), ss.max_date_list.items()):
                df_rft = determine_df_rft(df, max_date)
                ss.df_rft_list[bu] = df_rft

            ss.bgf_list = {}
            for bu, df_ch in ss.df_ch_list.items():
                bgf = bgf_fit(df_ch)
                ss.bgf_list[bu] = bgf

            bgf_evaluation_list_h = {}
            for (bu, bgf), (bu, df_ch) in zip(ss.bgf_list.items(), ss.df_ch_list.items()):
                bgf_eval = bgf_evaluation_prep_h(bgf, df_ch)
                bgf_evaluation_list_h[bu] = bgf_eval

            bgf_evaluation_list_a = {}
            for (bu, bgf), (bu, df_ch) in zip(ss.bgf_list.items(), ss.df_ch_list.items()):
                bgf_eval = bgf_evaluation_prep_a(bgf, df_ch)
                bgf_evaluation_list_a[bu] = bgf_eval

            ss.bgf_full_list = {}
            for bu, df_rft in ss.df_rft_list.items():
                bgf_full = bgf_full_fit(df_rft)
                ss.bgf_full_list[bu] = bgf_full

            # call helper function: predict each customer's purchases over multiple time periods
            t_FC = [90, 180, 270, 360]

            for t in t_FC:
                for (bu, df_rft), (bu, bgf) in zip(ss.df_rft_list.items(), ss.bgf_list.items()):
                    predict_purch(df_rft, bgf, t)

            for (bu,bgf), (bu, df_rft) in zip(ss.bgf_full_list.items(), ss.df_rft_list.items()):
                df_rft["prob_alive"] = calculate_prob_alive(bgf, df_rft)

            ss.df_rftv_list = {}
            for bu, df_rft in ss.df_rft_list.items():
                df_rftv = gg_prep(df_rft)
                ss.df_rftv_list[bu] = df_rftv

            ss.corr_list = {}
            for bu, df_rftv in ss.df_rftv_list.items():
                corr = gg_corr(df_rftv)
                ss.corr_list[bu] = corr

            ss.ggf_list = {}
            for bu, df_rftv in ss.df_rftv_list.items():
                ggf = gg_fit(df_rftv)
                ss.ggf_list[bu] = ggf

            for (bu, df_rftv), (bu, ggf) in zip(ss.df_rftv_list.items(), ss.ggf_list.items()):
                df_rftv_new = gg_avg(df_rftv, ggf)
                ss.df_rftv_list[bu] = df_rftv_new

            ss.mape_list = {}
            for bu, df_rftv in ss.df_rftv_list.items():
                mape = gg_evaluation(df_rftv)
                ss.mape_list[bu] = mape

            for (bu, df_rftv), (bu, ggf), (bu, bgf) in zip(ss.df_rftv_list.items(), ss.ggf_list.items(), ss.bgf_full_list.items()):
                df_rftv_new2 = compute_clv(df_rftv, ggf, bgf, annual_discount_rate, ss.expected_lifetime)
                ss.df_rftv_list[bu] = df_rftv_new2

            # Predicted vs Actual (Train-Test) Graph Container
            with st.container():
                st.write("Training BG/NBD Model: does the model reflect the actual data closely enough?")
                left_column, right_column = st.columns(2)
                with left_column:
                    tab_list = st.tabs(ss.bu_list)
                    for bu, tab in zip(ss.bu_list, tab_list):
                        with tab:
                            fig1 = eva_viz1(ss.bgf_list[bu])
                            st.pyplot(fig1.figure)

                with right_column:
                    tab_list = st.tabs(ss.bu_list)
                    for bu, tab in zip(ss.bu_list, tab_list):
                        with tab:
                            fig2 = plot_calibration_purchases_vs_holdout_purchases(ss.bgf_list[bu],
                                                                                   ss.df_ch_list[bu], n=9)
                            st.pyplot(fig2.figure)

            # Model Performance Metrics
            with st.container():
                left_column2, middle_column2, right_column2 = st.columns(3)
                with left_column2:
                    with st.expander("Performance Score"):
                        tab_list = st.tabs(ss.bu_list)
                        for bu, tab in zip(ss.bu_list, tab_list):
                            with tab:
                                st.write("BG/NBD Model Performance:")
                                st.markdown("* MAE: {0}".format(score_model(bgf_evaluation_list_a[bu],
                                                                            bgf_evaluation_list_h[bu],
                                                                            'mae')))
                                st.markdown("* RMSE: {0}".format(score_model(bgf_evaluation_list_a[bu],
                                                                            bgf_evaluation_list_h[bu],
                                                                             'rmse')))
                                st.write("Gamma-Gamma Model Performance:")
                                st.markdown("* Pearson correlation: %.3f" % ss.corr_list[bu])
                                st.markdown("* MAPE of predicted revenues: " + f'{ss.mape_list[bu]:.2f}')

                with middle_column2:
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

                with right_column2:
                    with st.expander("Prediction Difference of Calibration vs Holdout Purchases"):
                        tab_list = st.tabs(ss.bu_list)
                        for bu, tab in zip(ss.bu_list, tab_list):
                            with tab:

                                df_chi_square_test_cal_vs_hol, chi, pval, \
                                dof, exp, significance, critical_value = chi_square_test_customer_count(bu, ss.bgf_list)

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
