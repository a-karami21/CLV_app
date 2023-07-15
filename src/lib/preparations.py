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
def compute_clv(df_rftv, _ggf, _bgf, annual_discount_rate):
    DISCOUNT_a = annual_discount_rate  # annual discount rate
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