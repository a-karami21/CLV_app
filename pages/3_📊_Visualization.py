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
st.header("3. Visualization")
st.divider()

if ss.df_viz_list is None:
    st.subheader("Please Train the model in the Modelling page")
elif ss.df_viz_list is not None:
    # RFM Spread Visualization
    with st.container():
        st.subheader("RFM Distribution Visualization")
        tab_list = st.tabs(ss.product_list)
        for product, tab in zip(ss.product_list, tab_list):
            with tab:
                left_column, middle_column, right_column = st.columns(3)

                rfm_plot_dict = rfm_summary_figure(ss.df_rft_list, product)

                with left_column:
                    st.pyplot(rfm_plot_dict["Recency"].figure)
                with middle_column:
                    st.pyplot(rfm_plot_dict["Frequency"].figure)
                with right_column:
                    st.pyplot(rfm_plot_dict["Monetary"].figure)

    st.divider()

    # Customer P Alive Plot
    with st.container():
        st.subheader("Customer P(Alive) History Plot")
        tab_list = st.tabs(ss.product_list)
        for product, tab in zip(ss.product_list, tab_list):
            with tab:

                # Customer Selection
                customer_name_list = ss.df_viz_list[product]['Customer_Name']

                customer_selection = st.selectbox("Select customer ID to see its purchase behavior",
                                                  customer_name_list)

                selected_customer_id = ss.df_viz_list[product].loc[
                    ss.df_viz_list[product]["Customer_Name"] == customer_selection, 'Customer_ID'].values[0]
                selected_customer_name = ss.df_list[product][
                    ss.df_list[product]["Customer_ID"] == selected_customer_id]

                # P Alive History Plot & User Inputs
                with st.container():
                    p_alive_history_plot = fig_p_alive_history(product, ss.df_list, ss.bgf_full_list, selected_customer_name)
                    fig = st.pyplot(p_alive_history_plot.figure)

                # RFM & Last purchase date
                with st.container():
                    profile_col1, profile_col2, profile_col3, profile_col4, profile_col5, profile_col6 = st.columns(
                        6)

                    rfm_summary_dict = customer_rfm_summary(ss.df_viz_list, product, ss.df_list, selected_customer_id)

                    with profile_col1:
                        st.markdown("**Recency**")
                        st.markdown(str(rfm_summary_dict["Recency"]) + " Lifetime Length (Months)")
                    with profile_col2:
                        st.markdown("**Frequency**")
                        st.markdown(str(rfm_summary_dict["Frequency"]) + " Active Months")
                    with profile_col3:
                        st.markdown("**Monetary**")
                        st.markdown("$ " + f'{rfm_summary_dict["Monetary"]:,}' + " Average Purchase Value")
                    with profile_col4:
                        st.markdown("**Last Purchase Date**")
                        st.markdown(rfm_summary_dict["Last Purchase"])
                    with profile_col5:
                        st.markdown("**Predicted Purchase in Upcoming Year**")
                        st.markdown(str(rfm_summary_dict["Predicted Purchase"]) + " Purchases")
                    with profile_col6:
                        st.markdown("**CLV in Upcoming Year**")
                        st.markdown("$ " + f'{rfm_summary_dict["CLV"]:,}')

    # UI Divider
    st.divider()

    # Top 20 & Sales Growth Plot
    with st.container():
        df_plot_prep = df_plot_preparation(ss.df0, ss.merged_df)

        colors = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#EE8959", "#E76F51"]

        left_column, right_column = st.columns(2)
        with left_column:
            st.subheader("Top 20 Customers by CLV")

            industry_options = df_plot_prep["Industry"].unique().tolist()
            industry_options.insert(0, "All")

            industry_filter_selection = st.selectbox("Industry Filter", industry_options, key="top20_industry")
            product_filter_selection = st.multiselect("Product Filter", df_plot_prep["Product"].unique(), key="top20_product")

            df_plot = top_cust_data_prep(df_plot_prep, industry_filter_selection, product_filter_selection)

            st.altair_chart(fig_top20(df_plot, colors), use_container_width=True)

        with right_column:
            st.subheader("Predicted Sales Growth")

            # Filter Selection
            with st.container():
                left_column, right_column = st.columns(2)

                # Industry & Product Filter
                with left_column:

                    industry_options = df_plot_prep["Industry"].unique().tolist()
                    industry_options.insert(0, "All")

                    industry_filter_selection = st.selectbox("Industry Filter", industry_options,
                                                             key="growth_industry")
                    product_filter_selection = st.multiselect("Product Filter", df_plot_prep["Product"].unique(),
                                                              key="growth_product")
                # Customer Filter
                with right_column:
                    predicted_customer_only = st.checkbox("Predicted Customer Only?", value=True,
                                                          help="Filter Options only shows customer that have "
                                                               "value in FY2023 (Predicted)")

                    if not predicted_customer_only:
                        df_customer_filter = df_plot_prep
                    else:
                        df_customer_filter = df_plot_prep[df_plot_prep['Customer_Name'].isin(
                            df_plot_prep[df_plot_prep['Fiscal_Year'] == 'FY2023(Predicted)']['Customer_Name'])]

                    customer_filter_list = df_customer_filter["Customer_Name"].unique().tolist()
                    customer_filter_list.insert(0, "All")
                    customer_filter_selection = st.selectbox("Customer Filter", customer_filter_list,
                                                             key="growth_customer")

                    st.markdown("Note: The model only predicts customers that have repeated purchase"
                                " (more than once) so the predicted figure may be smaller than the previous year")


            df_plot = sales_growth_data_prep(df_plot_prep, industry_filter_selection,
                                             product_filter_selection, customer_filter_selection)

            plot = fig_sales_growth(df_plot, colors)
            st.altair_chart(plot, use_container_width=True)

    # UI Divider
    st.divider()

    # Line Plot for Comparison
    with st.container():
        st.subheader("Comparison Line Plot")
        left_column, right_column = st.columns((1,4))

        # Line Plot Filters
        with left_column:
            line_plot_group = st.selectbox("Select Grouping Variable", ["Industry", "Industry_Segment", "Product"],
                                           index=0, key="line_plot_group")

            industry_filter_selection = st.multiselect("Industry Filter", df_plot_prep["Industry"].unique(),
                                                       key="line_plot_industry")

            if not industry_filter_selection:
                df_industry_segment_selection = df_plot_prep
            else:
                df_industry_segment_selection = df_plot_prep[df_plot_prep['Industry'].isin(industry_filter_selection)]

            industry_segment_filter_selection = st.multiselect("Industry Segment Filter",
                                                               df_industry_segment_selection["Industry_Segment"].unique(),
                                                               key="line_plot_industry_segment")

            product_filter_selection = st.multiselect("Product Filter", df_plot_prep["Product"].unique(),
                                                      key="line_plot_product")

        # Line Plot Chart
        with right_column:
            df_plot = line_plot_data_prep(df_plot_prep, industry_filter_selection, industry_segment_filter_selection,
                                          product_filter_selection, line_plot_group)

            plot = fig_line_plot(df_plot, colors, line_plot_group)
            st.altair_chart(plot, use_container_width=True)

    # UI Divider
    st.divider()

    # Industry Segment
    with st.container():
        st.subheader("Industry Segment Treemap")

        # Filters Selection
        with st.container():
            left_column, right_column = st.columns(2)

            with left_column:
                year_filter_list = sorted(df_plot_prep["Fiscal_Year"].unique().tolist())
                year_filter_list.insert(0, "All")

                year_filter_selection = st.selectbox("Year Filter", year_filter_list,
                                                     key="treemap_year", index=0)

            with right_column:
                product_filter_list = df_plot_prep["Product"].unique().tolist()
                product_filter_list.insert(0, "All")

                product_filter_selection = st.selectbox("Product Filter", product_filter_list,
                                                        key="treemap_product",)

        df_plot = industry_treemap_data_prep(df_plot_prep, year_filter_selection, product_filter_selection)

        plot = fig_industry_treemap(df_plot)
        st.plotly_chart(plot, use_container_width=True)