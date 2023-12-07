from src.lib.preparations import *
from src.lib.visualizations import *
from src.lib.util import *

# initialize session state variables
ss = st.session_state

# Page Config
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# Side Bar & Main Panel Setup
col1 = st.sidebar
col2, col3 = st.columns((2, 1))

# Model Evaluation Section
st.header("2. Model Evaluation")

# Modelling User Input
with st.container():
    left_column, right_column = st.columns((1, 2))
    with left_column:
        # Interest Rate for Prediction Input
        annual_interest_rate = st.number_input("Annual Interest Rate (%) for One Year Prediction, Default is 6%",
                                               min_value=1.0, max_value=20.0, step=0.5, value=6.0) / 100

    st.divider()

    # Checkbox to Trigger Modelling Functions
    train_checkbox = st.checkbox("Train Model")

# Data Preparation
ss.product_list, ss.df_list = modelling_data_prep(ss.df0)

# Train BG/NBD to Fit on the full dataset
if train_checkbox:

    # Modelling Pipeline
    with st.container():
        modelling_output_dict = modelling_pipeline(ss.df_list, annual_interest_rate)

        ss.df_ch_list = modelling_output_dict["df_ch_list"]
        ss.max_date_list = modelling_output_dict["max_date_list"]
        ss.df_rft_list = modelling_output_dict["df_rft_list"]
        ss.bgf_list = modelling_output_dict["bgf_list"]
        ss.bgf_eval_list_predicted = modelling_output_dict["bgf_eval_list_predicted"]
        ss.bgf_eval_list_actual = modelling_output_dict["bgf_eval_list_actual"]
        ss.bgf_list = modelling_output_dict["bgf_list"]
        ss.bgf_full_list = modelling_output_dict["bgf_full_list"]
        ss.df_rftv_list = modelling_output_dict["df_rftv_list"]
        ss.corr_list = modelling_output_dict["corr_list"]
        ss.ggf_list = modelling_output_dict["ggf_list"]
        ss.mape_list = modelling_output_dict["mape_list"]

# Model Evaluation Visualization
    with st.expander("Model Evaluation Result", expanded=True):
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
                        st.markdown("* MAE: {0}".format(score_model(ss.bgf_eval_list_predicted[product],
                                                                    ss.bgf_eval_list_actual[product],
                                                                    'mae')))
                        st.markdown("* RMSE: {0}".format(score_model(ss.bgf_eval_list_predicted[product],
                                                                    ss.bgf_eval_list_actual[product],
                                                                     'rmse')))
                        st.write("Gamma-Gamma Model Performance:")
                        st.markdown("* Pearson correlation: %.3f" % ss.corr_list[product])
                        st.markdown("* MAPE of predicted revenues: " + f'{ss.mape_list[product]:.2f}')

                    # Chi-Square Test (Customer Count)
                    with middle_column:
                        st.write("Prediction Difference of Customer Count")

                        df_chi_square_test_customer_count, chi, pval,\
                        dof, exp, significance, critical_value = chi_square_test_customer_count(product, ss.bgf_list)

                        st.dataframe(df_chi_square_test_customer_count.style.format("{:,.0f}"))

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

                        st.dataframe(df_chi_square_test_cal_vs_hol.style.format("{:,.2f}"))

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







