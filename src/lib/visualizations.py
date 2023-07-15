import pandas as pd
import streamlit as st

from lifetimes.plotting import \
    plot_period_transactions, \
    plot_calibration_purchases_vs_holdout_purchases, \
    plot_history_alive

import matplotlib.pyplot as plt
import altair as alt
import plotly.express as px
import seaborn as sns

import math

from scipy.stats import chi2_contingency
from scipy.stats import chi2

# Predicted vs Actual (Train-Test) Graph Container
@st.cache_data
def evaluation_visualization(_bgf, df_ch):

    # Plot 1: Number of Customer each Period Level
    n = _bgf.data.shape[0]
    simulated_data = _bgf.generate_new_data(size=n)

    model_counts = pd.DataFrame(_bgf.data["frequency"].value_counts().sort_index().iloc[:9])

    simulated_counts = pd.DataFrame(
        simulated_data["frequency"].value_counts().sort_index().iloc[:9])

    combined_counts = model_counts.merge(simulated_counts, how="outer", left_index=True,
                                         right_index=True).fillna(0)

    combined_counts.columns = ["Actual", "Model"]

    fig1 = combined_counts.plot(kind="bar")

    fig1.bar_label(fig1.containers[0], label_type='edge')
    fig1.bar_label(fig1.containers[1], label_type='edge')

    plt.legend()
    plt.title("Frequency of Repeat Transactions")
    plt.ylabel("Customers")
    plt.xlabel("Number of Calibration Period Transactions")

    # Plot 2: Calibration vs Holdout Purchases & Model vs Actual
    fig2 = plot_calibration_purchases_vs_holdout_purchases(_bgf, df_ch, n=9)

    evaluation_plot_dict = {"Plot1": fig1, "Plot2": fig2}

    return evaluation_plot_dict

# Chisquare Dataframe
@st.cache_data
def chi_square_test_customer_count(bu, _bgf_list):
    n = _bgf_list[bu].data.shape[0]
    simulated_data = _bgf_list[bu].generate_new_data(size=n)

    model_counts = pd.DataFrame(_bgf_list[bu].data["frequency"].value_counts().sort_index().iloc[:])
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

@st.cache_data
def chi_square_test_cal_vs_hol(df_ch_list, _bgf_list, bu):
    kind = "frequency_cal"

    summary = df_ch_list[bu].copy()
    duration_holdout = summary.iloc[0]["duration_holdout"]

    summary["model_predictions"] = _bgf_list[bu].conditional_expected_number_of_purchases_up_to_time(
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


@st.cache_data
def export_table(df_rftv, df):
    df_final = df_rftv

    df_final = df_final.reset_index().merge(df[["Customer_ID",
                                                "Customer_Name",
                                                "Industry",
                                                "Industry_Segment"]],
                                            how="left").set_index('Customer_ID')

    order = ['Customer_Name', 'Industry','Industry_Segment', 'CLV', 'frequency', 'recency', 'T', 'monetary_value',
             'predict_purch_90', 'predict_purch_180', 'predict_purch_270', 'predict_purch_360',
             'prob_alive', 'exp_avg_rev', 'avg_rev', 'error_rev']

    df_final = df_final[order]
    df_final = df_final.drop_duplicates().reset_index()

    return df_final

@st.cache_resource
def convert_df(df):
    return df.to_csv().encode('utf-8')

@st.cache_data
def rfm_summary_figure(df_rft_list, product):
    max_freq_product = df_rft_list[product]["frequency"].quantile(0.98)
    max_rec_product = df_rft_list[product]["recency"].max()
    max_mon_product = df_rft_list[product]["monetary_value"].quantile(0.98)
    max_T_product = df_rft_list[product]["T"].max()

    # training recency
    fig1 = plt.figure(figsize=(5, 5))
    ax1 = sns.distplot(df_rft_list[product]["recency"])
    ax1.set_xlim(0, max_rec_product)
    ax1.set_title("recency (days): distribution of the customers")

    # training: frequency
    fig2 = plt.figure(figsize=(5, 5))
    ax2 = sns.distplot(df_rft_list[product]["frequency"])
    ax2.set_xlim(0, max_freq_product)
    ax2.set_title("frequency (days): distribution of the customers")

    # training: monetary
    fig3 = plt.figure(figsize=(5, 5))
    ax3 = sns.distplot(df_rft_list[product]["monetary_value"])
    ax3.set_xlim(0, max_mon_product)
    ax3.set_title("monetary (USD): distribution of the customers")
    # st.pyplot(fig3.figure)

    rfm_plot_dict = {"Recency": fig1, "Frequency": fig2, "Monetary": fig3}

    return rfm_plot_dict


@st.cache_data
def fig_p_alive_history(product, df_list, _bgf_full_list, selected_customer):
    # selected customer: cumulative transactions
    max_date = df_list[product]["Transaction_Date"].max()
    min_date = df_list[product]["Transaction_Date"].min()
    span_days = (max_date - min_date).days

    # history of the selected customer: probability over time of being alive
    plt.figure(figsize=(20, 4))
    fig = plot_history_alive(model=_bgf_full_list[product],
                              t=span_days,
                              transactions=selected_customer,
                              datetime_col="Transaction_Date")

    return fig


@st.cache_data
def customer_rfm_summary(df_viz_list, product, df_list, selected_customer_id):
    df_rfm_customer = df_viz_list[product][df_viz_list[product]["Customer_ID"] == selected_customer_id]

    max_date = df_list[product]["Transaction_Date"].max()

    customer_recency = math.ceil((df_rfm_customer.iloc[0]['recency']) / 30)
    customer_frequency = math.ceil(df_rfm_customer.iloc[0]['frequency'])
    customer_monetary = math.ceil(df_rfm_customer.iloc[0]['monetary_value'])
    cust_last_purchase_date = max_date.date()
    predicted_purchase = math.ceil(df_rfm_customer.iloc[0]['predict_purch_360'])
    customer_CLV = math.ceil(df_rfm_customer.iloc[0]['CLV'])

    rfm_summary_dict = {"Recency": customer_recency, "Frequency": customer_frequency,
                        "Monetary": customer_monetary, "Last Purchase": cust_last_purchase_date,
                        "Predicted Purchase": predicted_purchase, "CLV": customer_CLV}

    return rfm_summary_dict


@st.cache_data
def df_plot_preparation(df0, df_predicted):
    grouped_df0 = df0.groupby(['Customer_ID', 'Customer_Name', 'Fiscal_Year',
                               'Product', 'Industry', 'Industry_Segment']
                              )['Transaction_Value'].sum().reset_index()

    column_order = ['Customer_ID', 'Customer_Name', 'Product',
                    'Industry', 'Industry_Segment', 'CLV']

    df_plot_prep = df_predicted[column_order]
    df_plot_prep = df_plot_prep.rename(columns={'CLV': 'Transaction_Value'})
    df_plot_prep = df_plot_prep.drop(df_plot_prep.columns[6:], axis=1)
    df_plot_prep.insert(3, "Fiscal_Year", "FY2023(Predicted)")

    df_plot_prep = pd.concat([grouped_df0, df_plot_prep]).reset_index(drop=True)

    return df_plot_prep


@st.cache_data
def top_cust_data_prep(df_plot_prep, industry_filter, product_filter_list):
    # Filter df to Predicted Year
    df_filtered = df_plot_prep[df_plot_prep["Fiscal_Year"] == "FY2023(Predicted)"]

    # Filter df to selected industry
    if industry_filter == "All":
        pass
    else:
        df_filtered = df_filtered[df_filtered["Industry"] == industry_filter]

    # Filter df to selected products
    if not product_filter_list:
        pass
    else:
        df_filtered = df_filtered[df_filtered["Product"].isin(product_filter_list)]

    # Group Customer by transaction value
    df_grouped = df_filtered.groupby('Customer_Name')['Transaction_Value'].sum().reset_index()
    df_grouped.columns = ['Customer_Name', 'Total Value']

    # Add the Rank column based on descending order of total value
    df_grouped['Rank'] = df_grouped['Total Value'].rank(ascending=False).astype(int)

    # Merge the rank information back to the original DataFrame
    df_plot = pd.merge(df_filtered, df_grouped[['Customer_Name', 'Rank']], on='Customer_Name')

    return df_plot


@st.cache_data()
def fig_top20(df_plot, colors):
    plot = alt.Chart(df_plot).mark_bar().encode(
        y=alt.Y('Customer_Name', stack=True, sort=alt.EncodingSortField(field="Rank", op='min', order='ascending')),
        x='Transaction_Value',
        color=alt.Color('Product', scale=alt.Scale(range=colors)),
    ).transform_filter(alt.datum.Rank <= 20).properties(height=550).interactive()

    return plot

@st.cache_data
def sales_growth_data_prep(df_plot_prep, industry_filter, product_filter_list, customer_filter):

    df_plot = df_plot_prep

    # Filter df to selected industry
    if industry_filter == "All":
        pass
    else:
        df_plot = df_plot[df_plot["Industry"] == industry_filter]

    # Filter df to selected products
    if not product_filter_list:
        pass
    else:
        df_plot = df_plot[df_plot["Product"].isin(product_filter_list)]

    # Filter df to selected customer

    if customer_filter == "All":
        pass
    else:
        df_plot = df_plot[df_plot["Customer_Name"] == customer_filter]

    df_plot = df_plot.groupby(["Fiscal_Year", "Product"])['Transaction_Value'].sum().reset_index()

    return df_plot

@st.cache_data()
def fig_sales_growth(df_plot, colors):
    plot = alt.Chart(df_plot).mark_bar().encode(
        x='Fiscal_Year',
        y='Transaction_Value',
        color=alt.Color('Product', scale=alt.Scale(range=colors))

    ).properties(height=500).interactive()

    return plot

@st.cache_data()
def line_plot_data_prep(df_plot_prep, industry_filter, industry_segment_filter, product_filter, group_selection):
    df_plot = df_plot_prep

    # Filter df to selected industry
    if not industry_filter:
        pass
    else:
        df_plot = df_plot[df_plot["Industry"].isin(industry_filter)]

    # Filter df to selected industry
    if not industry_segment_filter:
        pass
    else:
        df_plot = df_plot[df_plot["Industry_Segment"].isin(industry_segment_filter)]

    # Filter df to selected products
    if not product_filter:
        pass
    else:
        df_plot = df_plot[df_plot["Product"].isin(product_filter)]

    if group_selection == "Industry":
        df_plot = df_plot.groupby(["Fiscal_Year", "Industry"])['Transaction_Value'].sum().reset_index()
    elif group_selection == "Industry_Segment":
        df_plot = df_plot.groupby(["Fiscal_Year", "Industry_Segment"])['Transaction_Value'].sum().reset_index()
    elif group_selection == "Product":
        df_plot = df_plot.groupby(["Fiscal_Year", "Product"])['Transaction_Value'].sum().reset_index()

    return df_plot

@st.cache_data()
def fig_line_plot(df_plot, colors, group):
    plot = alt.Chart(df_plot).mark_line(point=alt.OverlayMarkDef(color="red")).encode(
        x='Fiscal_Year',
        y='Transaction_Value',
        color=alt.Color(group, scale=alt.Scale(range=colors))

    ).properties(height=500).interactive()

    text = alt.Chart(df_plot).mark_text(dy=15, color='black', fontSize=15).encode(
        x='Fiscal_Year',
        y='Transaction_Value',
        color=alt.Color(group, scale=alt.Scale(range=colors)),
        text=alt.Text("Transaction_Value:Q", format=',.0f')
    )

    plot = alt.layer(plot, text).configure_point(size=50)

    return plot

@st.cache_data()
def industry_treemap_data_prep(df_plot_prep, year_filter, product_filter):
    df_plot = df_plot_prep

    # Filter df to selected year
    if year_filter == "All":
        pass
    else:
        df_plot = df_plot[df_plot["Fiscal_Year"] == year_filter]

    # Filter df to selected product
    if product_filter == "All":
        pass
    else:
        df_plot = df_plot[df_plot["Product"] == product_filter]

    df_plot = df_plot.groupby(['Industry', 'Industry_Segment'])['Transaction_Value'].sum().reset_index()

    return df_plot

@st.cache_data()
def fig_industry_treemap(df_plot):
    # Industry Segment Treemap
    plot = px.treemap(df_plot,
                      path=[px.Constant('All'), 'Industry', 'Industry_Segment'],
                      values='Transaction_Value',
                      height=500)

    plot.update_traces(textinfo="label+percent root+percent parent")

    plot.update_layout(
        treemapcolorway=["orange", "darkblue", "green"],
        margin=dict(t=0, l=0, r=0, b=0)
    )

    return plot