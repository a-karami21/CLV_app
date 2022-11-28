import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import seaborn as sns
import streamlit as st
import io
import plotly.express as px
import plotly.graph_objects as go
import math

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource

from sklearn.metrics import mean_absolute_percentage_error

from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import \
    calibration_and_holdout_data, \
    summary_data_from_transaction_data

from lifetimes.plotting import \
    plot_period_transactions, \
    plot_history_alive, \
    plot_calibration_purchases_vs_holdout_purchases

from bokeh.models import FactorRange

sns.set(rc={'image.cmap': 'coolwarm'})

# Initialization
ss = st.session_state

# Page Config
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

col1 = st.sidebar
col2, col3 = st.columns((2,1))

if ss.df_viz is None:
    st.header("Please train the model in the Homepage")
else:
    col1.header("3. Output Filters")
    with col1.expander("Select Filter"):
        industry_type_cust = st.selectbox("Select Industry Type for Top 20 Customers",
                                          ('Energy', 'Material', 'Life', 'All'), index=3)

        p_alive_slider = st.slider("Probability alive lower than X %", 10, 100, 80)
        ss.prob_alive_input = float(p_alive_slider / 100)

    st.header("Visualization")

    # RFM Spread Visualization
    with st.expander("RFM Stats"):
        st.write("RFM Spread Visualization")
        left_column3, middle_column3, right_column3 = st.columns(3)

        # training: axis length
        max_freq = ss.df_rft["frequency"].quantile(0.98)
        max_rec = ss.df_rft["recency"].max()
        max_mon = ss.df_rft["monetary_value"].quantile(0.98)
        max_T = ss.df_rft["T"].max()

        with left_column3:
            # training recency
            fig4 = plt.figure(figsize=(5, 5))
            ax = sns.distplot(ss.df_rft["recency"])
            ax.set_xlim(0, max_rec)
            ax.set_title("recency (days): distribution of the customers")
            st.pyplot(fig4.figure)
        with middle_column3:
            # training: frequency
            fig5 = plt.figure(figsize=(5, 5))
            ax = sns.distplot(ss.df_rft["frequency"])
            ax.set_xlim(0, max_freq)
            ax.set_title("frequency (days): distribution of the customers")
            st.pyplot(fig5.figure)
        with right_column3:
            # training: monetary
            fig6 = plt.figure(figsize=(5, 5))
            ax = sns.distplot(ss.df_rft["monetary_value"])
            ax.set_xlim(0, max_mon)
            ax.set_title("monetary (USD): distribution of the customers")
            st.pyplot(fig6.figure)

    # Customer Behavior Simulation
    with st.container():
        st.subheader("Customer P(Alive) History")

        # Customer Selection
        if ss.df_viz is not None:
            customers_name = ss.df_viz['ec_eu_customer_n']
            customer_name_input = st.selectbox("Select customer ID to see its purchase behavior", customers_name)
            ss.customer_name_input = customer_name_input
            customer_id_input = ss.df_viz.loc[ss.df_viz.ec_eu_customer_n == customer_name_input, 'ec_eu_customer'].values[0]

            ss.customer_id_input = customer_id_input

        # P Alive History
        custID = ss.customer_id_input

        df1C = ss.df1[ss.df1["ec_eu_customer"] == custID]

        # X selected customer: cumulative transactions
        max_date = ss.df1["order_intake_date"].max()
        min_date = ss.df1["order_intake_date"].min()
        span_days = (max_date - min_date).days

        # history of the selected customer: probability over time of being alive
        st.write(ss.customer_name_input)
        plt.figure(figsize=(20 ,4))
        fig7 = plot_history_alive(model=ss.bgf,
                                    t=span_days,
                                    transactions=df1C,
                                    datetime_col="order_intake_date"
                                    )

        st.pyplot(fig7.figure)

        # RFM & Last purchase date
        with st.container():
            profile_col1, profile_col2, profile_col3, profile_col4, profile_col5, profile_col6 = st.columns(6)

            df_rfm_customer = ss.df_viz[ss.df_viz["ec_eu_customer"] == custID]

            with profile_col1:
                st.markdown("**Recency**")

                customer_recency = math.ceil((df_rfm_customer.iloc[0]['recency'])/30)
                st.markdown(str(customer_recency) + " Lifetime Length")

            with profile_col2:
                st.markdown("**Frequency**")

                customer_frequency = math.ceil(df_rfm_customer.iloc[0]['frequency'])
                st.markdown(str(customer_frequency) + " Active Months")

            with profile_col3:
                st.markdown("**Monetary**")

                customer_monetary = math.ceil(df_rfm_customer.iloc[0]['monetary_value'])
                st.markdown("$ " + f'{customer_monetary:,}' + " Average Purchase Value")

            with profile_col4:
                st.markdown("**Last Purchase Date**")
                cust_last_purchase_date = max_date.date()
                st.markdown(cust_last_purchase_date)

            with profile_col5:
                st.markdown("**Predicted Purchase**")

                customer_purchase = math.ceil(df_rfm_customer.iloc[0]['predict_purch_360'])
                st.markdown(str(customer_purchase) + " Purchases")

            with profile_col6:
                st.markdown("**CLV**")

                customer_CLV = math.ceil(df_rfm_customer.iloc[0]['CLV'])
                st.markdown("$ " + f'{customer_CLV:,}')


    # Top 20 Customer
    with st.container():
        st.subheader("CLV Prediction on " + str(ss.expected_lifetime) + " Months in the future." )
        viz_left_column1, viz_right_column1 = st.columns(2)
        with viz_left_column1:
            st.markdown("Top 20 Customers by CLV")

            if industry_type_cust == "Energy":
                df_top20 = ss.df_viz[ss.df_viz['ec_eu_industry_type_n'].isin(['Energy'])]
            elif industry_type_cust == "Material":
                df_top20 = ss.df_viz[ss.df_viz['ec_eu_industry_type_n'].isin(['Material'])]
            elif industry_type_cust == "Life":
                df_top20 = ss.df_viz[ss.df_viz['ec_eu_industry_type_n'].isin(['Life'])]
            elif industry_type_cust == "All":
                df_top20 = ss.df_viz

            df_top20 = df_top20[:20].reset_index()
            y = df_top20['ec_eu_customer_n']
            x = df_top20['CLV' ] /1000

            graph = figure(y_range=list(reversed(y)),
                            plot_height=500,
                            toolbar_location= None
                            )

            graph.hbar(y=y, right=x, height=0.5, fill_color="#ffcc66", line_color="black")

            graph.xaxis.axis_label = "CLV (K USD)"

            st.bokeh_chart(graph)

        with viz_right_column1:
            st.markdown("Top 20 Customers with P Alive < " +str(ss.prob_alive_input*100) +" %")

            if industry_type_cust == "Energy":
                df_p_alive = ss.df_viz[ss.df_viz['ec_eu_industry_type_n'].isin(['Energy'])]
            elif industry_type_cust == "Material":
                df_p_alive = ss.df_viz[ss.df_viz['ec_eu_industry_type_n'].isin(['Material'])]
            elif industry_type_cust == "Life":
                df_p_alive = ss.df_viz[ss.df_viz['ec_u_industry_type_n'].isin(['Life'])]
            elif industry_type_cust == "All":
                df_p_alive = ss.df_viz

            df_p_alive = df_p_alive[ss.df_viz["prob_alive"] < (ss.prob_alive_input)].reset_index()
            df_p_alive = df_p_alive.iloc[:20]

            y2 = df_p_alive['ec_eu_customer_n']
            x2 = df_p_alive['CLV'] / 1000

            graph2 = figure(y_range=list(reversed(y2)),
                                plot_height=500,
                                toolbar_location= None
                                )

            graph2.hbar(y=y2, right=x2, height=0.5, fill_color="grey", line_color="black")

            graph2.xaxis.axis_label = "CLV (K USD)"

            st.bokeh_chart(graph2)

    # Industry Segment
    with st.container():
        st.subheader("Industry Type & Segment CLV")
        viz_left_column2, viz_right_column2 = st.columns((2,1))

        df_industry_viz = ss.df_viz.groupby(['ec_eu_industry_type_n',
                                             'ec_eu_industry_sub_segment_n'])['CLV'].mean().sort_values(
            ascending=False).reset_index()

        # Industry Segment Treemap
        with viz_left_column2:
            st.markdown("Industry Segment Treemap")

            fig = px.treemap(df_industry_viz,
                             path = [px.Constant('All'), 'ec_eu_industry_type_n', 'ec_eu_industry_sub_segment_n'],
                             values = 'CLV',
                             width = 760,
                             height = 400)

            fig.update_layout(
                treemapcolorway= ["orange", "darkblue", "green"],
                margin = dict(t=0, l=0, r=0, b=0)
            )

            st.plotly_chart(fig)

        # Top 10 Industry Segment
        with viz_right_column2:

            df_industry_viz = df_industry_viz[:10].reset_index()

            st.markdown("Top 10 Industry Segment")
            y = df_industry_viz['ec_eu_industry_sub_segment_n']
            x = df_industry_viz['CLV'] / 1000

            graph3 = figure(y_range=list(reversed(y)),
                            toolbar_location=None,
                            plot_height=400,
                            plot_width=400,
                            )

            graph3.hbar(y=y, right=x, height=0.5 ,fill_color="#ff9966", line_color="black")

            graph3.xaxis.axis_label = "CLV (K USD)"

            st.bokeh_chart(graph3)
