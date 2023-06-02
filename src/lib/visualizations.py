import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from scipy.stats import chi2_contingency
from scipy.stats import chi2

# Predicted vs Actual (Train-Test) Graph Container
def eva_viz1(bgf):
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

# Chisquare Dataframe
def chi_square_test_customer_count(bu, bgf_list):
    n = bgf_list[bu].data.shape[0]
    simulated_data = bgf_list[bu].generate_new_data(size=n)

    model_counts = pd.DataFrame(bgf_list[bu].data["frequency"].value_counts().sort_index().iloc[:])
    simulated_counts = pd.DataFrame(simulated_data["frequency"].value_counts().sort_index().iloc[:])
    combined_counts = model_counts.merge(simulated_counts, how="outer", left_index=True,
                                             right_index=True).fillna(0)
    combined_counts.columns = ["Actual", "Model"]

    df_chi_square_test_customer_count = combined_counts
    df_chi_square_test_customer_count['Difference'] = combined_counts['Actual'] - combined_counts['Model']

    df_chi_square_test_customer_count.index.name = "Transaction"

    df_chi_square_test_customer_count_9 = df_chi_square_test_customer_count[8:]
    df_chi_square_test_customer_count_9 = df_chi_square_test_customer_count_9.mean(axis=0)
    df_chi_square_test_customer_count_9.name = "9+"

    df_chi_square_test_customer_count = df_chi_square_test_customer_count[:8].append(df_chi_square_test_customer_count_9)

    chi, pval, dof, exp = chi2_contingency(df_chi_square_test_customer_count.iloc[:, :2])

    significance = 0.001
    p = 1 - significance
    critical_value = chi2.ppf(p, dof)

    return df_chi_square_test_customer_count, chi, pval, dof, exp, significance, critical_value

def chi_square_test_cal_vs_hol(df_ch_list, bgf_list, bu):
    kind = "frequency_cal"

    summary = df_ch_list[bu].copy()
    duration_holdout = summary.iloc[0]["duration_holdout"]

    summary["model_predictions"] = bgf_list[bu].conditional_expected_number_of_purchases_up_to_time(
        duration_holdout, summary["frequency_cal"], summary["recency_cal"], summary["T_cal"])

    purch_diff = summary.groupby(kind)[["frequency_holdout", "model_predictions"]].mean().iloc[:]

    purch_diff['Difference'] = purch_diff['model_predictions'] - purch_diff['frequency_holdout']

    df_chi_square_test_cal_vs_hol = purch_diff

    df_chi_square_test_cal_vs_hol.index.name = "Transaction"

    df_chi_square_test_cal_vs_hol_9 = df_chi_square_test_cal_vs_hol[8:]
    df_chi_square_test_cal_vs_hol_9 = df_chi_square_test_cal_vs_hol_9.mean(axis=0)
    df_chi_square_test_cal_vs_hol_9.name = "9+"

    df_chi_square_test_cal_vs_hol = df_chi_square_test_cal_vs_hol[:8].append(df_chi_square_test_cal_vs_hol_9)

    chi, pval, dof, exp = chi2_contingency(df_chi_square_test_cal_vs_hol.iloc[:, :2])

    significance = 0.001
    p = 1 - significance
    critical_value = chi2.ppf(p, dof)

    return df_chi_square_test_cal_vs_hol, chi, pval, dof, exp, significance, critical_value


def export_table(df_rftv, df):
    df_final = df_rftv

    df_final = df_final.reset_index().merge(df[["ec_eu_customer",
                                                "ec_eu_customer_n",
                                                "ec_eu_industry_type_n",
                                                "ec_eu_industry_sub_segment_n",
                                                "ec_eu_industry_segment_n"]],
                                            how="left").set_index('ec_eu_customer')

    order = ['ec_eu_customer_n', 'ec_eu_industry_type_n', 'ec_eu_industry_sub_segment_n',
             'ec_eu_industry_segment_n', 'CLV', 'frequency', 'recency', 'T', 'monetary_value',
             'predict_purch_90', 'predict_purch_180', 'predict_purch_270', 'predict_purch_360',
             'prob_alive', 'exp_avg_rev', 'avg_rev', 'error_rev']

    df_final = df_final[order]
    df_final = df_final.drop_duplicates()
    df_final = df_final.rename_axis('ec_eu_customer').reset_index()

    return df_final

@st.cache_resource
def convert_df(df):
    return df.to_csv().encode('utf-8')
