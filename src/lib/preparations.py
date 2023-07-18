import pandas as pd
import numpy as np
import streamlit as st
from datetime import timedelta

from sklearn.metrics import mean_absolute_percentage_error

from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import \
    calibration_and_holdout_data, \
    summary_data_from_transaction_data

@st.cache_data
def read_order_intake_csv(dataset_file):
    df0 = pd.read_csv(dataset_file, sep=";")
    return df0

def modelling_data_prep(df0):

    # Treat CustomerID as a categorical variable
    df0["Customer_ID"].astype(np.int64).astype(object)
    # Convert Date Column to Datetime Format
    df0["Transaction_Date"] = pd.to_datetime(df0["Transaction_Date"])
    # Convert Transaction Value Column to Numeric Format
    df0["Transaction_Value"] = pd.to_numeric(df0["Transaction_Value"])

    # Create Dictionary for Each Product Category & Its Dataframe
    product_list = df0["Product"].unique().tolist()
    df_list = {}

    for product in product_list:
        df = df0[df0["Product"].isin([product])]
        df_list[product] = df

    return product_list, df_list


# train/test split (calibration/holdout)
def train_test_split(df):
    t_holdout = 365                                         # days to reserve for holdout period

    max_date = df["Transaction_Date"].max()                     # end date of observations

    max_cal_date = max_date - timedelta(days=t_holdout)     # end date of chosen calibration period

    df_ch = calibration_and_holdout_data(
            transactions=df,
            customer_id_col="Customer_ID",
            datetime_col="Transaction_Date",
            monetary_value_col="Transaction_Value",
            calibration_period_end=max_cal_date,
            observation_period_end=max_date,
            freq='D',
            )
    return df_ch

def max_date_func(df):
    max_date = df["Transaction_Date"].max()
    return max_date


# determine recency, frequency, T, monetary value for each customer
def determine_df_rft(df, max_date):
    df_rft = summary_data_from_transaction_data(
        transactions=df,
        customer_id_col="Customer_ID",
        datetime_col="Transaction_Date",
        monetary_value_col="Transaction_Value",
        observation_period_end=max_date,
        freq="D")

    return df_rft

# Fit the BG/NBD Model
def bgf_fit(df_ch):
    bgf = BetaGeoFitter(penalizer_coef=1e-06)
    bgf.fit(
        frequency=df_ch["frequency_cal"],
        recency=df_ch["recency_cal"],
        T=df_ch["T_cal"])

    return bgf

def bgf_eval_prep_predicted(bgf, df_ch):
    # get predicted frequency during holdout period
    frequency_holdout_predicted = bgf.predict(df_ch['duration_holdout'],
                                              df_ch['frequency_cal'],
                                              df_ch['recency_cal'],
                                              df_ch['T_cal'])

    return frequency_holdout_predicted


def bgf_eval_prep_actual(bgf, df_ch):
    # get actual frequency during holdout period
    frequency_holdout_actual = df_ch['frequency_holdout']

    return frequency_holdout_actual


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


def predict_purch(df, bgf, t):
    df["predict_purch_" + str(t)] = \
        bgf.predict(
            t,
            df["frequency"],
            df["recency"],
            df["T"])

# probability that a customer is alive for each customer in dataframe
def calculate_prob_alive(bgf, df_rft):
    prob_alive = bgf.conditional_probability_alive(
        frequency=df_rft["frequency"],
        recency=df_rft["recency"],
        T=df_rft["T"])

    return prob_alive

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

# fitting the Gamma-Gamma model
@st.cache_resource
def gg_fit(df_rftv):
    ggf = GammaGammaFitter(penalizer_coef=0.001)
    ggf.fit(
        frequency=df_rftv["frequency"],
        monetary_value=df_rftv["monetary_value"])

    return ggf

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

# compute customer lifetime value
@st.cache_resource
def compute_clv(df_rftv, _ggf, _bgf, annual_interest_rate):
    DISCOUNT_a = annual_interest_rate  # annual discount rate
    LIFE = 12 # lifetime expected for the customers in months

    discount_m = (1 + DISCOUNT_a) ** (1 / 12) - 1  # monthly discount rate

    clv = _ggf.customer_lifetime_value(
        transaction_prediction_model=_bgf,
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


@st.cache_resource
def modelling_pipeline(df_list, annual_interest_rate):

    # BG/NBD Modelling
    #   Train test split
    df_ch_list = {}
    for product, df in df_list.items():
        df_ch = train_test_split(df)
        df_ch_list[product] = df_ch

    #   Finding max date
    max_date_list = {}
    for product, df in df_list.items():
        max_date = max_date_func(df)
        max_date_list[product] = max_date

    #   RFM aggregation dataframe
    df_rft_list = {}
    for (product, df), (product, max_date) in zip(df_list.items(), max_date_list.items()):
        df_rft = determine_df_rft(df, max_date)
        df_rft_list[product] = df_rft

    #   Fitting BG/NBD
    bgf_list = {}
    for product, df_ch in df_ch_list.items():
        bgf = bgf_fit(df_ch)
        bgf_list[product] = bgf

    # Model Evaluation
    #   Predicted Frequency
    bgf_eval_list_predicted = {}
    for (product, bgf), (product, df_ch) in zip(bgf_list.items(), df_ch_list.items()):
        bgf_eval = bgf_eval_prep_predicted(bgf, df_ch)
        bgf_eval_list_predicted[product] = bgf_eval

    #   Actual Frequency
    bgf_eval_list_actual = {}
    for (product, bgf), (product, df_ch) in zip(bgf_list.items(), df_ch_list.items()):
        bgf_eval = bgf_eval_prep_actual(bgf, df_ch)
        bgf_eval_list_actual[product] = bgf_eval

    #   Train model to the full dataset
    bgf_full_list = {}
    for product, df_rft in df_rft_list.items():
        bgf_full = bgf_full_fit(df_rft)
        bgf_full_list[product] = bgf_full

    #   Predicting future purchase
    #   call helper function: predict each customer's purchases over multiple time periods
    t_FC = [90, 180, 270, 360]

    for t in t_FC:
        for (product, df_rft), (product, bgf) in zip(df_rft_list.items(), bgf_list.items()):
            predict_purch(df_rft, bgf, t)

    #   Calculate probability alive
    for (product, bgf), (product, df_rft) in zip(bgf_full_list.items(), df_rft_list.items()):
        df_rft["prob_alive"] = calculate_prob_alive(bgf, df_rft)

    #   Create new dataframe (added predicted future purchase & prob alive)
    df_rftv_list = {}
    for product, df_rft in df_rft_list.items():
        df_rftv = gg_prep(df_rft)
        df_rftv_list[product] = df_rftv

    # Gamma-Gamma Modelling
    #   Calculate F & M Pearson Correlation
    corr_list = {}
    for product, df_rftv in df_rftv_list.items():
        corr = gg_corr(df_rftv)
        corr_list[product] = corr

    #   Gamma-Gamma Model Fitting
    ggf_list = {}
    for product, df_rftv in df_rftv_list.items():
        ggf = gg_fit(df_rftv)
        ggf_list[product] = ggf

    #   Estimate average error of the predicted monetary value
    for (product, df_rftv), (product, ggf) in zip(df_rftv_list.items(), ggf_list.items()):
        df_rftv_new = gg_avg(df_rftv, ggf)
        df_rftv_list[product] = df_rftv_new

    #   Calculate MAPE for Gamma-Gamma evaluation
    mape_list = {}
    for product, df_rftv in df_rftv_list.items():
        mape = gg_evaluation(df_rftv)
        mape_list[product] = mape

    #   Predict Customer Lifetime Value and add to the output dataframe
    for (product, df_rftv), (product, ggf), (product, bgf) in zip(df_rftv_list.items(), ggf_list.items(),
                                                                  bgf_full_list.items()):
        df_rftv_new2 = compute_clv(df_rftv, ggf, bgf, annual_interest_rate)
        df_rftv_list[product] = df_rftv_new2

    modelling_output_dict = {"df_ch_list": df_ch_list, "max_date_list": max_date_list, "df_rft_list": df_rft_list,
                             "df_rft_list": df_rft_list, "bgf_list": bgf_list, "bgf_eval_list_predicted": bgf_eval_list_predicted,
                             "bgf_eval_list_actual": bgf_eval_list_actual, "bgf_full_list": bgf_full_list,
                             "df_rftv_list": df_rftv_list, "corr_list": corr_list, "ggf_list": ggf_list, "mape_list": mape_list}
    
    return modelling_output_dict