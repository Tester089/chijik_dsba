import io
import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import requests
import streamlit as st

window = st.sidebar.radio(
    "Section",
    ["Introduction", "Dataset Description", "Data Cleanup", "Stats", "Hypothesis", "Conclusion"]
)

API_BASE = os.getenv("API_BASE")


@st.cache_data(ttl=3600, show_spinner=False)
def _load_df():
    try:
        response = requests.get(f"{API_BASE}/dataset", timeout=300)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        st.error("ConnectionError")
        st.stop()
    except requests.exceptions.HTTPError as exc:
        st.error("HTTPError")
        st.stop()

    df = pd.read_csv(io.StringIO(response.text))

    return df


if window == "Introduction":
    st.markdown("""

    * Egor Dumalkin 252-2
    * Matvey Veber 252-1




    The project analyzes a store-level dataset related to X5 Group / Pyaterochka. The main goal of the work is to explore store turnover data, check data quality, identify anomalies, create useful derived features, visualize important patterns, and test hypotheses about store performance. The report follows the full exploratory data analysis process: dataset inspection, cleanup, descriptive statistics, general and detailed visual overview, data transformation, hypothesis testing, and final discussion.

    The project was completed collaboratively by both team members. Egor Dumalkin mainly focused on the technical implementation of the analytical part, including data preprocessing, exploratory data analysis, descriptive statistics, feature engineering, visualizations, and hypothesis testing. Matvey Veber contributed to the analytical design of the project: together we searched for data anomalies, selected the most relevant plots, formulated the hypotheses, interpreted the results, and prepared the final conclusions. Matvey was also mainly responsible for developing the web interface, integrating the results into the project page, and preparing the final interactive version of the report.

    """)


elif window == "Dataset Description":
    st.markdown("""
    The dataset contains monthly observations for Pyaterochka stores. Each row represents one store in one month. The main target variable is `РТО`, which represents monthly retail turnover. Other fields describe store characteristics, local environment indicators, calendar fields, and categorical descriptors.

    The notebook uses the local file `train_2.csv`. If the file is not available locally, the loading cell can download it from Google Drive.


    """)

    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (12, 5)

    if "df" not in st.session_state:
        st.session_state.df = _load_df()
    df = st.session_state.df

    st.markdown("### First rows of the dataset")
    st.dataframe(df.head())
    st.markdown("### Column types")
    st.dataframe(df.dtypes.reset_index())

    n_rows, n_cols = df.shape
    n_stores = df["new_id"].nunique()
    period_summary = df[["Год", "Месяц"]].drop_duplicates().sort_values(["Год", "Месяц"])
    rows_per_store = df.groupby("new_id").size()

    overview = pd.DataFrame({
        "Metric": [
            "Number of rows",
            "Number of columns",
            "Number of unique stores",
            "Number of unique year-month periods",
            "Minimum year",
            "Maximum year",
            "Minimum rows per store",
            "Maximum rows per store",
            "Median rows per store",
        ],
        "Value": [
            n_rows,
            n_cols,
            n_stores,
            period_summary.shape[0],
            df["Год"].min(),
            df["Год"].max(),
            rows_per_store.min(),
            rows_per_store.max(),
            rows_per_store.median(),
        ],
    })

    st.markdown("### Bazirovanie stats about the dataset")
    st.dataframe(overview)

    stores_by_period = pd.crosstab(df["Год"], df["Месяц"])

    st.markdown("### How many stores we have in year-month")
    st.dataframe(stores_by_period)

    st.markdown(""" 
    """)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    column_types_summary = pd.DataFrame({
        "Feature type": ["Numeric columns", "Categorical or text columns"],
        "Count": [len(numeric_cols), len(categorical_cols)],
        "Columns": [numeric_cols, categorical_cols],
    })

    st.markdown("""### Numeric and categorial columns""")
    st.dataframe(column_types_summary)

    st.markdown(""" 
    """)

    important_fields = pd.DataFrame({
        "Column": [
            "new_id",
            "Год", "Месяц",
            "РТО",
            "Рабочие часы в день",
            "Количество касс",
            "Трафик пеший, в час",
            "Трафик авто, в час",
            "Среднее количество товаров в чеке",
            "Среднее количество промо товаров ",
            "Торговая площадь, категориальный",
            "Дата открытия, категориальный",
            "Регион",
            "Населенный пункт",
        ],
        "Meaning": [
            "Store identifier",
            "Observation year",
            "Observation month",
            "Monthly retail turnover",
            "Average working hours per day",
            "Number of cash registers",
            "Pedestrian traffic per hour near the store",
            "Car traffic per hour near the store",
            "Average number of items in a receipt",
            "Average number of promotional items in a receipt",
            "Categorical store area group",
            "Categorical store age group",
            "Russian region",
            "Settlement or city",
        ]
    })
    st.markdown("### Definition of the columns")
    st.dataframe(important_fields)



elif window == "Data Cleanup":
    st.markdown("""
    The dataset has no explicit NaN values, but this does not mean that the data is clean. Some missing or incorrect values are encoded as zeros or physically impossible numbers. This section checks explicit missing values, data types, suspicious values, and the cleanup procedure.
    """)

    if "df" not in st.session_state:
        st.session_state.df = _load_df()
    df = st.session_state.df

    missing_before = (
        df.isna()
        .sum()
        .reset_index()
        .rename(columns={"index": "column", 0: "missing_values"})
    )
    missing_before["missing_share"] = missing_before["missing_values"] / len(df)

    st.markdown("There`re no NaNs)")
    st.dataframe(missing_before.sort_values("missing_values", ascending=False))

    st.markdown(""" 
    """)

    dtype_summary = pd.DataFrame({
        "column": df.columns,
        "dtype": [df[col].dtype for col in df.columns],
        "unique_values": [df[col].nunique() for col in df.columns],
    })
    st.markdown("""How many unique values in each data type""")
    st.dataframe(dtype_summary)

    st.markdown(""" 
    """)

    rules = {
        "work_hours_gt_24": df["Рабочие часы в день"] > 24,
        "work_hours_lt_6": df["Рабочие часы в день"] < 6,
        "cash_registers_zero": df["Количество касс"] == 0,
        "population_zero": df["Численность населения"] == 0,
        "households_zero": df["Количество домохозяйств"] == 0,
        "both_traffic_zero": (df["Трафик пеший, в час"] == 0) & (df["Трафик авто, в час"] == 0),
        "pedestrian_traffic_zero": df["Трафик пеший, в час"] == 0,
        "car_traffic_zero": df["Трафик авто, в час"] == 0,
        "promo_items_gt_total_items": (
                df["Среднее количество промо товаров "] >
                df["Среднее количество товаров в чеке"]
        ),
    }

    anomaly_summary = pd.DataFrame({
        "rows": {name: mask.sum() for name, mask in rules.items()},
        "stores": {name: df.loc[mask, "new_id"].nunique() for name, mask in rules.items()},
        "share": {name: mask.mean() for name, mask in rules.items()},
    })
    st.markdown("### Suspicious founded values ")
    st.dataframe(anomaly_summary.sort_values("rows", ascending=False))

    st.markdown(""" 
    """)

    rto_p90 = df["РТО"].quantile(0.90)
    zero_traffic_high_rto = df[
        (df["Трафик пеший, в час"] == 0) &
        (df["Трафик авто, в час"] == 0) &
        (df["РТО"] >= rto_p90)
        ]

    st.markdown("""### Stores with zero traffic but high turnover""")
    st.dataframe(zero_traffic_high_rto[[
        "new_id", "Год", "Месяц",
        "Трафик пеший, в час", "Трафик авто, в час",
        "РТО", "Населенный пункт", "Регион"
    ]].sort_values("РТО", ascending=False).head(10))

    st.markdown(""" 
    """)

    num_cols_for_iqr = df.select_dtypes(include="number").columns.drop(
        ["new_id", "Год", "Месяц", "Флаг алкогольной лицензии"]
    )

    iqr_report = []
    for col in num_cols_for_iqr:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (df[col] < lower) | (df[col] > upper)
        iqr_report.append({
            "column": col,
            "lower": lower,
            "upper": upper,
            "outlier_rows": mask.sum(),
            "outlier_share": mask.mean(),
        })

    st.markdown("### Outliers based on the IQR rule")
    st.dataframe(pd.DataFrame(iqr_report).sort_values("outlier_rows", ascending=False))

    st.markdown(""" 
    """)

    df["rto_store_median"] = df.groupby("new_id")["РТО"].transform("median")
    df["rto_to_store_median"] = df["РТО"] / df["rto_store_median"]

    rto_spikes = df[df["rto_to_store_median"] > 4]
    rto_drops = df[df["rto_to_store_median"] < 0.25]

    st.markdown("### Stores with extremely high or low turnover compared to their own usual level")
    st.dataframe(pd.DataFrame({
        "case": ["RTO > 4 times store median", "RTO < 0.25 times store median"],
        "rows": [len(rto_spikes), len(rto_drops)],
        "stores": [rto_spikes["new_id"].nunique(), rto_drops["new_id"].nunique()],
    }))

    st.markdown(""" 
    """)

    clean = df.copy()

    clean["is_bad_work_hours"] = clean["Рабочие часы в день"] > 24
    clean["is_low_work_hours"] = clean["Рабочие часы в день"] < 6
    clean["is_zero_population"] = clean["Численность населения"] == 0
    clean["is_zero_households"] = clean["Количество домохозяйств"] == 0
    clean["is_zero_both_traffic"] = (
            (clean["Трафик пеший, в час"] == 0) &
            (clean["Трафик авто, в час"] == 0)
    )
    clean["is_zero_cash_registers"] = clean["Количество касс"] == 0

    clean.loc[clean["Рабочие часы в день"] > 24, "Рабочие часы в день"] = np.nan
    clean.loc[clean["Рабочие часы в день"] < 6, "Рабочие часы в день"] = np.nan
    clean.loc[clean["Численность населения"] == 0, "Численность населения"] = np.nan
    clean.loc[clean["Количество домохозяйств"] == 0, "Количество домохозяйств"] = np.nan
    clean.loc[clean["Количество касс"] == 0, "Количество касс"] = np.nan

    traffic_zero = (
            (clean["Трафик пеший, в час"] == 0) &
            (clean["Трафик авто, в час"] == 0)
    )
    clean.loc[traffic_zero, ["Трафик пеший, в час", "Трафик авто, в час"]] = np.nan

    st.markdown(""" 
    """)


    def fill_by_group_median(data, col, group_levels):
        result = data[col].copy()
        for group_cols in group_levels:
            group_median = data.groupby(group_cols)[col].transform("median")
            result = result.fillna(group_median)
        return result.fillna(data[col].median())


    clean["Рабочие часы в день"] = fill_by_group_median(
        clean,
        "Рабочие часы в день",
        [["Регион", "Торговая площадь, категориальный"], ["Торговая площадь, категориальный"], ["Регион"]]
    )

    clean["Численность населения"] = fill_by_group_median(
        clean,
        "Численность населения",
        [["Регион", "Населенный пункт"], ["Регион"]]
    )

    clean["Количество домохозяйств"] = fill_by_group_median(
        clean,
        "Количество домохозяйств",
        [["Регион", "Населенный пункт"], ["Регион"]]
    )

    clean["Количество касс"] = fill_by_group_median(
        clean,
        "Количество касс",
        [["Регион", "Торговая площадь, категориальный"], ["Торговая площадь, категориальный"]]
    )

    for col in ["Трафик пеший, в час", "Трафик авто, в час"]:
        clean[col] = fill_by_group_median(
            clean,
            col,
            [["Регион", "Населенный пункт", "Торговая площадь, категориальный"], ["Регион", "Населенный пункт"],
             ["Регион"]]
        )

    st.session_state.clean = clean

    cleanup_check = pd.DataFrame({
        "check": [
            "Remaining explicit NaN values",
            "Working hours below 6",
            "Working hours above 24",
            "Zero cash registers",
            "Zero population",
            "Both traffic values equal zero",
        ],
        "rows": [
            clean.isna().sum().sum(),
            (clean["Рабочие часы в день"] < 6).sum(),
            (clean["Рабочие часы в день"] > 24).sum(),
            (clean["Количество касс"] == 0).sum(),
            (clean["Численность населения"] == 0).sum(),
            ((clean["Трафик пеший, в час"] == 0) & (clean["Трафик авто, в час"] == 0)).sum(),
        ]
    })
    st.markdown("### After cleanup")
    st.dataframe(cleanup_check)

    st.markdown("### Final column types after cleanup")
    st.dataframe(clean.dtypes.reset_index().rename(columns={"index": "column", 0: "dtype"}))

elif window == "Stats":
    st.markdown(""" 

    This section provides descriptive statistics for the most important numerical fields. The table includes mean, median, and standard deviation.

    """)

    if "clean" not in st.session_state:
        st.error("Cleanup")
        st.stop()
    clean = st.session_state.clean

    stats_cols = [
        "РТО",
        "Количество касс",
        "Трафик пеший, в час",
        "Трафик авто, в час",
        "Рабочие часы в день",
    ]

    st.markdown("### Descriptive statistics for key numerical columns")
    descriptive_stats = clean[stats_cols].agg(["mean", "median", "std"]).T
    st.dataframe(descriptive_stats)

    st.markdown(""" 

    This section gives a first visual overview of the cleaned dataset. It includes histograms, boxplots, line plots, bar plots, and traffic comparisons.



    Turnover is a monetary variable and usually has a long right tail. We compare the original turnover distribution with a logarithmic transformation used only for visualization.

    """)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    sns.histplot(clean["РТО"], bins=60, kde=True, ax=axes[0])
    axes[0].set_title("Distribution of Monthly Turnover")
    axes[0].set_xlabel("РТО")
    axes[0].set_ylabel("Number of observations")
    sns.histplot(np.log1p(clean["РТО"]), bins=60, kde=True, ax=axes[1], color="teal")
    axes[1].set_title("Distribution of Log Monthly Turnover")
    axes[1].set_xlabel("log(РТО + 1)")
    axes[1].set_ylabel("Number of observations")
    plt.tight_layout()
    st.pyplot(plt.gcf())

    st.markdown(""" 
    """)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    sns.boxplot(x=clean["РТО"], ax=axes[0])
    axes[0].set_title("Boxplot of Monthly Turnover")
    axes[0].set_xlabel("РТО")
    sns.boxplot(x=np.log1p(clean["РТО"]), ax=axes[1], color="teal")
    axes[1].set_title("Boxplot of Log Monthly Turnover")
    axes[1].set_xlabel("log(РТО + 1)")
    plt.tight_layout()
    st.pyplot(plt.gcf())

    st.markdown(""" 
    The next plots compare monthly mean and median turnover by year. This helps identify seasonality and differences between years.
    """)

    monthly_turnover = (
        clean.groupby(["Год", "Месяц"])["РТО"]
        .agg(mean="mean", median="median")
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    sns.lineplot(data=monthly_turnover, x="Месяц", y="median", hue="Год", marker="o", ax=axes[0])
    axes[0].set_title("Median Monthly Turnover by Year")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Median РТО")
    axes[0].set_xticks(range(1, 13))
    sns.lineplot(data=monthly_turnover, x="Месяц", y="mean", hue="Год", marker="o", ax=axes[1])
    axes[1].set_title("Mean Monthly Turnover by Year")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Mean РТО")
    axes[1].set_xticks(range(1, 13))
    plt.tight_layout()
    st.pyplot(plt.gcf())

    st.markdown(""" 
    """)

    stores_by_period_plot = clean.groupby(["Год", "Месяц"])["new_id"].nunique().reset_index()
    stores_by_period_plot["period"] = stores_by_period_plot["Год"].astype(str) + "-" + stores_by_period_plot[
        "Месяц"].astype(str).str.zfill(2)
    baseline_stores = stores_by_period_plot["new_id"].iloc[0]
    stores_by_period_plot["difference_from_first_period"] = stores_by_period_plot["new_id"] - baseline_stores

    plt.figure(figsize=(16, 5))
    sns.barplot(data=stores_by_period_plot, x="period", y="difference_from_first_period", color="steelblue")
    plt.axhline(0, color="black", linewidth=1)
    plt.title(f"Change in Number of Stores by Month Compared with First Period (baseline = {baseline_stores})")
    plt.xlabel("Period")
    plt.ylabel("Difference from first period")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt.gcf())

    st.markdown(""" 
    The following plots show the distribution of categorical store descriptors and several important numerical fields.
    """)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    sns.countplot(data=clean, x="Торговая площадь, категориальный",
                  order=clean["Торговая площадь, категориальный"].value_counts().index, ax=axes[0])
    axes[0].set_title("Store Area Category Counts")
    axes[0].set_xlabel("Store area category")
    axes[0].set_ylabel("Number of observations")
    axes[0].tick_params(axis="x", rotation=20)
    sns.countplot(data=clean, x="Дата открытия, категориальный",
                  order=clean["Дата открытия, категориальный"].value_counts().index, ax=axes[1])
    axes[1].set_title("Store Age Category Counts")
    axes[1].set_xlabel("Store age category")
    axes[1].set_ylabel("Number of observations")
    axes[1].tick_params(axis="x", rotation=20)
    plt.tight_layout()
    st.pyplot(plt.gcf())

    st.markdown(""" 
    """)

    total_traffic_plot = clean["Трафик пеший, в час"] + clean["Трафик авто, в час"]
    numeric_overview = pd.DataFrame({
        "Рабочие часы в день": clean["Рабочие часы в день"],
        "Количество касс": clean["Количество касс"],
        "Трафик пеший, в час": clean["Трафик пеший, в час"],
        "Трафик авто, в час": clean["Трафик авто, в час"],
        "Общий трафик": total_traffic_plot,
        "Среднее количество товаров в чеке": clean["Среднее количество товаров в чеке"],
    })

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    axes = axes.flatten()
    for ax, col in zip(axes, numeric_overview.columns):
        sns.histplot(numeric_overview[col], bins=50, kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Number of observations")
    plt.tight_layout()
    st.pyplot(plt.gcf())

    st.markdown(""" 
    """)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.boxplot(x=clean["Рабочие часы в день"], ax=axes[0])
    axes[0].set_title("Working Hours per Day")
    axes[0].set_xlabel("Hours")
    sns.boxplot(x=clean["Количество касс"], ax=axes[1])
    axes[1].set_title("Number of Cash Registers")
    axes[1].set_xlabel("Cash registers")
    sns.boxplot(x=total_traffic_plot, ax=axes[2])
    axes[2].set_title("Total Traffic")
    axes[2].set_xlabel("Pedestrian + car traffic")
    plt.tight_layout()
    st.pyplot(plt.gcf())

    st.markdown(""" 
    """)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    sns.histplot(clean["Трафик пеший, в час"], bins=50, kde=True, ax=axes[0], label="Pedestrian traffic")
    sns.histplot(clean["Трафик авто, в час"], bins=50, kde=True, ax=axes[0], color="orange", label="Car traffic",
                 alpha=0.6)
    axes[0].set_title("Pedestrian and Car Traffic Distributions")
    axes[0].set_xlabel("Traffic per hour")
    axes[0].set_ylabel("Number of observations")
    axes[0].legend()

    traffic_sample = clean.sample(n=min(20000, len(clean)), random_state=42)
    sns.scatterplot(
        data=traffic_sample,
        x="Трафик пеший, в час",
        y="Трафик авто, в час",
        hue="Торговая площадь, категориальный",
        alpha=0.35,
        s=20,
        ax=axes[1]
    )
    axes[1].set_title("Pedestrian vs Car Traffic")
    axes[1].set_xlabel("Pedestrian traffic per hour")
    axes[1].set_ylabel("Car traffic per hour")
    axes[1].legend(title="Store area", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    st.pyplot(plt.gcf())

    st.markdown(""" 
    The next plots summarize regional representation and the frequency of anomaly flags created during cleanup.
    """)

    region_store_count = (
        clean.groupby("Регион")["new_id"]
        .nunique()
        .sort_values(ascending=False)
        .head(15)
        .reset_index(name="stores")
    )
    region_median_rto = (
        clean.groupby("Регион")["РТО"]
        .median()
        .sort_values(ascending=False)
        .head(15)
        .reset_index(name="median_RTO")
    )

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.barplot(data=region_store_count, y="Регион", x="stores", ax=axes[0], color="steelblue")
    axes[0].set_title("Top 15 Regions by Number of Stores in Dataset")
    axes[0].set_xlabel("Number of stores")
    axes[0].set_ylabel("Region")
    sns.barplot(data=region_median_rto, y="Регион", x="median_RTO", ax=axes[1], color="seagreen")
    axes[1].set_title("Top 15 Regions by Median Turnover")
    axes[1].set_xlabel("Median РТО")
    axes[1].set_ylabel("Region")
    plt.tight_layout()
    st.pyplot(plt.gcf())

    st.markdown(""" 
    """)

    flag_cols = [
        "is_bad_work_hours",
        "is_low_work_hours",
        "is_zero_population",
        "is_zero_households",
        "is_zero_both_traffic",
        "is_zero_cash_registers",
    ]
    flag_summary = (
        clean[flag_cols]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "flag", 0: "share"})
    )

    plt.figure(figsize=(12, 5))
    sns.barplot(data=flag_summary, x="share", y="flag", color="salmon")
    plt.title("Share of Observations with Anomaly Flags")
    plt.xlabel("Share of observations")
    plt.ylabel("Anomaly flag")
    plt.tight_layout()
    st.pyplot(plt.gcf())

    st.markdown("### Share of rows flagged as anomalous")
    st.dataframe(flag_summary)

    st.markdown("""
    This section explores more detailed relationships between variables. It includes regional comparisons, grouped turnover distributions, correlations, scatter plots, and categorical interactions.


    The dataset contains a large share of Pyaterochka stores, but we do not have official store counts by region inside the dataset. Therefore, regional store counts below describe the available dataset and should be interpreted carefully.

    """)

    estimated_total_pyaterochka_stores_2023 = 21_308
    stores_in_dataset = clean["new_id"].nunique()
    approx_dataset_coverage = stores_in_dataset / estimated_total_pyaterochka_stores_2023

    st.markdown("### How much of Pyaterochka's network is in our data?")
    st.dataframe(pd.DataFrame({
        "Metric": [
            "Stores in dataset",
            "Estimated total Pyaterochka stores in Russia, 2023",
            "Approximate national dataset coverage",
        ],
        "Value": [
            stores_in_dataset,
            estimated_total_pyaterochka_stores_2023,
            approx_dataset_coverage,
        ]
    }))

    detailed = clean.assign(
        total_traffic_temp=clean["Трафик пеший, в час"] + clean["Трафик авто, в час"],
        log_RTO_temp=np.log1p(clean["РТО"]),
    )

    region_analysis = (
        detailed.groupby("Регион")
        .agg(
            stores_in_dataset=("new_id", "nunique"),
            observations=("new_id", "size"),
            total_RTO_in_dataset=("РТО", "sum"),
            mean_RTO=("РТО", "mean"),
            median_RTO=("РТО", "median"),
            median_population=("Численность населения", "median"),
            median_total_traffic=("total_traffic_temp", "median"),
            median_cash_registers=("Количество касс", "median"),
            big_store_share=("Торговая площадь, категориальный", lambda x: x.isin(["Большой", "Очень большой"]).mean()),
            alcohol_license_share=("Флаг алкогольной лицензии", "mean"),
            bad_population_share=("is_zero_population", "mean"),
            zero_traffic_share=("is_zero_both_traffic", "mean"),
        )
        .reset_index()
    )

    region_analysis["avg_monthly_RTO_per_store"] = (
            region_analysis["total_RTO_in_dataset"] / region_analysis["observations"]
    )

    st.markdown("### Top regions by total turnover")
    region_analysis_filtered = region_analysis[region_analysis["stores_in_dataset"] >= 30].copy()
    st.dataframe(region_analysis_filtered.sort_values("total_RTO_in_dataset", ascending=False).head(15))

    st.markdown(""" 
    """)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    top_total = region_analysis_filtered.sort_values("total_RTO_in_dataset", ascending=False).head(15)
    sns.barplot(data=top_total, y="Регион", x="total_RTO_in_dataset", ax=axes[0, 0], color="darkcyan")
    axes[0, 0].set_title("Top Regions by Total Turnover in Dataset")
    axes[0, 0].set_xlabel("Total РТО in dataset")
    axes[0, 0].set_ylabel("Region")
    top_median = region_analysis_filtered.sort_values("median_RTO", ascending=False).head(15)
    sns.barplot(data=top_median, y="Регион", x="median_RTO", ax=axes[0, 1], color="seagreen")
    axes[0, 1].set_title("Top Regions by Median Store-Month Turnover")
    axes[0, 1].set_xlabel("Median РТО")
    axes[0, 1].set_ylabel("Region")
    top_big_share = region_analysis_filtered.sort_values("big_store_share", ascending=False).head(15)
    sns.barplot(data=top_big_share, y="Регион", x="big_store_share", ax=axes[1, 0], color="coral")
    axes[1, 0].set_title("Top Regions by Share of Big Stores")
    axes[1, 0].set_xlabel("Share of big and very big stores")
    axes[1, 0].set_ylabel("Region")
    top_population = region_analysis_filtered.sort_values("median_population", ascending=False).head(15)
    sns.barplot(data=top_population, y="Регион", x="median_population", ax=axes[1, 1], color="steelblue")
    axes[1, 1].set_title("Top Regions by Median Population Around Stores")
    axes[1, 1].set_xlabel("Median population value")
    axes[1, 1].set_ylabel("Region")
    plt.tight_layout()
    st.pyplot(plt.gcf())

    st.markdown(""" 
    """)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.scatterplot(
        data=region_analysis_filtered,
        x="stores_in_dataset",
        y="total_RTO_in_dataset",
        size="median_RTO",
        hue="big_store_share",
        sizes=(60, 700),
        alpha=0.75,
        palette="viridis",
        ax=axes[0],
    )
    axes[0].set_title("Total Regional Turnover vs Number of Stores in Dataset")
    axes[0].set_xlabel("Stores in dataset")
    axes[0].set_ylabel("Total РТО in dataset")
    for _, row in region_analysis_filtered.nlargest(6, "total_RTO_in_dataset").iterrows():
        axes[0].annotate(row["Регион"], (row["stores_in_dataset"], row["total_RTO_in_dataset"]), fontsize=8)
    sns.scatterplot(
        data=region_analysis_filtered,
        x="median_population",
        y="median_RTO",
        size="stores_in_dataset",
        hue="big_store_share",
        sizes=(60, 700),
        alpha=0.75,
        palette="viridis",
        ax=axes[1],
    )
    axes[1].set_xscale("log")
    axes[1].set_title("Median Turnover vs Median Population Around Stores")
    axes[1].set_xlabel("Median population value, log scale")
    axes[1].set_ylabel("Median РТО")
    plt.tight_layout()
    st.pyplot(plt.gcf())

    st.markdown(""" 

    Aggregate regional statistics can hide variation between stores. The following boxplot compares store-month turnover distributions in the largest regions.

    """)

    top_regions_by_stores = (
        region_analysis
        .sort_values("stores_in_dataset", ascending=False)
        .head(12)["Регион"]
    )

    plt.figure(figsize=(16, 8))
    sns.boxplot(
        data=detailed[detailed["Регион"].isin(top_regions_by_stores)],
        y="Регион",
        x="log_RTO_temp",
        order=top_regions_by_stores,
    )
    plt.title("Log Turnover Distribution in the Largest Regions by Store Count")
    plt.xlabel("log(РТО + 1)")
    plt.ylabel("Region")
    plt.tight_layout()
    st.pyplot(plt.gcf())

    st.markdown(""" 

    A correlation matrix and scatter plots help identify relationships between numerical features and turnover.

    """)

    corr_data = clean.assign(
        log_RTO_temp=np.log1p(clean["РТО"]),
        total_traffic_temp=clean["Трафик пеший, в час"] + clean["Трафик авто, в час"],
        promo_share_temp=np.where(
            clean["Среднее количество товаров в чеке"] > 0,
            clean["Среднее количество промо товаров "] / clean["Среднее количество товаров в чеке"],
            np.nan,
        ),
    )

    corr_cols = [
        "log_RTO_temp",
        "Рабочие часы в день",
        "Количество касс",
        "total_traffic_temp",
        "Трафик пеший, в час",
        "Трафик авто, в час",
        "Среднее количество товаров в чеке",
        "Среднее количество промо товаров ",
        "promo_share_temp",
        "Среднее количество отмен",
        "Численность населения",
        "Количество домохозяйств",
        "Флаг алкогольной лицензии",
    ]

    corr_matrix = corr_data[corr_cols].corr(method="spearman")

    plt.figure(figsize=(14, 11))
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0, linewidths=0.5)
    plt.title("Spearman Correlation Heatmap")
    plt.tight_layout()
    st.pyplot(plt.gcf())

    st.markdown("### Spearman correlation with log turnover")
    st.dataframe(corr_matrix["log_RTO_temp"].sort_values(ascending=False))

    st.markdown(""" 
    """)

    plot_sample = corr_data.sample(n=min(30000, len(corr_data)), random_state=42)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    sns.scatterplot(data=plot_sample, x="Количество касс", y="log_RTO_temp", hue="Торговая площадь, категориальный",
                    alpha=0.35, s=20, ax=axes[0, 0])
    axes[0, 0].set_title("Log Turnover vs Number of Cash Registers")
    axes[0, 0].set_xlabel("Cash registers")
    axes[0, 0].set_ylabel("log(РТО + 1)")
    sns.scatterplot(data=plot_sample, x="total_traffic_temp", y="log_RTO_temp", hue="Торговая площадь, категориальный",
                    alpha=0.35, s=20, ax=axes[0, 1])
    axes[0, 1].set_title("Log Turnover vs Total Traffic")
    axes[0, 1].set_xlabel("Total traffic")
    axes[0, 1].set_ylabel("log(РТО + 1)")
    sns.scatterplot(data=plot_sample, x="Рабочие часы в день", y="log_RTO_temp", hue="Торговая площадь, категориальный",
                    alpha=0.35, s=20, ax=axes[1, 0])
    axes[1, 0].set_title("Log Turnover vs Working Hours")
    axes[1, 0].set_xlabel("Working hours per day")
    axes[1, 0].set_ylabel("log(РТО + 1)")
    sns.scatterplot(data=plot_sample, x="promo_share_temp", y="log_RTO_temp", hue="Торговая площадь, категориальный",
                    alpha=0.35, s=20, ax=axes[1, 1])
    axes[1, 1].set_title("Log Turnover vs Promo Share")
    axes[1, 1].set_xlabel("Promo share in average receipt")
    axes[1, 1].set_ylabel("log(РТО + 1)")

    plt.tight_layout()
    st.pyplot(plt.gcf())

    st.markdown(""" 
    """)

    plt.figure(figsize=(14, 6))
    sns.boxplot(
        data=corr_data,
        x="Торговая площадь, категориальный",
        y="log_RTO_temp",
        hue="Флаг алкогольной лицензии",
        order=corr_data.groupby("Торговая площадь, категориальный")["log_RTO_temp"].median().sort_values().index,
    )
    plt.title("Log Turnover by Store Area and Alcohol License")
    plt.xlabel("Store area category")
    plt.ylabel("log(РТО + 1)")
    plt.legend(title="Alcohol license")
    plt.tight_layout()
    st.pyplot(plt.gcf())

elif window == "Hypothesis":
    if "clean" not in st.session_state:
        st.error("Cleanup")
        st.stop()
    clean = st.session_state.clean

    if "total_traffic" not in clean.columns:
        clean["total_traffic"] = clean["Трафик пеший, в час"] + clean["Трафик авто, в час"]
    if "promo_share" not in clean.columns:
        clean["promo_share"] = np.where(
            clean["Среднее количество товаров в чеке"] > 0,
            clean["Среднее количество промо товаров "] / clean["Среднее количество товаров в чеке"],
            np.nan,
        )
    if "rto_per_cash_register" not in clean.columns:
        clean["rto_per_cash_register"] = np.where(
            clean["Количество касс"] > 0,
            clean["РТО"] / clean["Количество касс"],
            np.nan,
        )

    st.markdown("""

    This section tests two hypotheses. The goal is to evaluate whether the patterns in the dataset support the hypotheses.



    **Hypothesis:** same-store turnover growth in grocery retail is higher than official inflation.

    This hypothesis is interesting because `РТО` is a monetary variable observed over several years. We compare January 2024 with January 2023 and January 2025 with January 2024 for the same stores. January is used because it is available for all three years.

    Important limitation: `РТО` is not a price index. Its growth can be caused by price inflation, more purchases, product mix, promotions, or customer behavior.


    """)

    official_inflation = {
        "Jan 2023 -> Jan 2024": 0.0742,
        "Jan 2024 -> Jan 2025": 0.0952,
    }

    january_rto = (
        clean[(clean["Месяц"] == 1) & (clean["Год"].isin([2023, 2024, 2025]))]
        .pivot(index="new_id", columns="Год", values="РТО")
        .dropna(subset=[2023, 2024, 2025])
    )

    january_rto["growth_2024_vs_2023"] = january_rto[2024] / january_rto[2023] - 1
    january_rto["growth_2025_vs_2024"] = january_rto[2025] / january_rto[2024] - 1

    inflation_comparison = pd.DataFrame({
        "period": ["Jan 2023 -> Jan 2024", "Jan 2024 -> Jan 2025"],
        "official_inflation": [
            official_inflation["Jan 2023 -> Jan 2024"],
            official_inflation["Jan 2024 -> Jan 2025"],
        ],
        "median_same_store_RTO_growth": [
            january_rto["growth_2024_vs_2023"].median(),
            january_rto["growth_2025_vs_2024"].median(),
        ],
        "mean_same_store_RTO_growth": [
            january_rto["growth_2024_vs_2023"].mean(),
            january_rto["growth_2025_vs_2024"].mean(),
        ],
        "share_of_stores_above_official_inflation": [
            (january_rto["growth_2024_vs_2023"] > official_inflation["Jan 2023 -> Jan 2024"]).mean(),
            (january_rto["growth_2025_vs_2024"] > official_inflation["Jan 2024 -> Jan 2025"]).mean(),
        ],
    })

    inflation_comparison["median_growth_minus_official_inflation"] = (
            inflation_comparison["median_same_store_RTO_growth"] - inflation_comparison["official_inflation"]
    )

    st.markdown("### Same-store January turnover growth vs official inflation")
    st.dataframe(inflation_comparison)

    inflation_plot_data = inflation_comparison.melt(
        id_vars="period",
        value_vars=["official_inflation", "median_same_store_RTO_growth", "mean_same_store_RTO_growth"],
        var_name="metric",
        value_name="growth_rate",
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(data=inflation_plot_data, x="period", y="growth_rate", hue="metric")
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Same-Store January Turnover Growth vs Official Inflation")
    plt.xlabel("Period")
    plt.ylabel("Growth rate")
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    plt.legend(title="Metric")
    plt.tight_layout()
    st.pyplot(plt.gcf())

    growth_2024 = january_rto["growth_2024_vs_2023"]
    growth_2025 = january_rto["growth_2025_vs_2024"]

    growth_long = pd.DataFrame({
        "Jan 2024 vs Jan 2023": growth_2024,
        "Jan 2025 vs Jan 2024": growth_2025,
    }).melt(var_name="period", value_name="growth_rate")

    x_min = growth_long["growth_rate"].quantile(0.05)
    x_max = growth_long["growth_rate"].quantile(0.95)

    fig, axes = plt.subplots(2, 1, figsize=(16, 11))

    sns.ecdfplot(data=growth_long, x="growth_rate", hue="period", ax=axes[0])
    axes[0].axvline(official_inflation["Jan 2023 -> Jan 2024"], color="red", linestyle="--",
                    label="Official inflation 2023")
    axes[0].axvline(official_inflation["Jan 2024 -> Jan 2025"], color="darkred", linestyle="--",
                    label="Official inflation 2024")
    axes[0].set_xlim(x_min, x_max)
    axes[0].set_title("Cumulative Distribution of Same-Store January RTO Growth")
    axes[0].set_xlabel("Growth rate")
    axes[0].set_ylabel("Share of stores")
    axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    axes[0].legend()

    sns.histplot(
        data=growth_long[(growth_long["growth_rate"] >= x_min) & (growth_long["growth_rate"] <= x_max)],
        x="growth_rate",
        hue="period",
        bins=50,
        kde=True,
        element="step",
        stat="density",
        common_norm=False,
        ax=axes[1],
    )
    axes[1].axvline(official_inflation["Jan 2023 -> Jan 2024"], color="red", linestyle="--",
                    label="Official inflation 2023")
    axes[1].axvline(official_inflation["Jan 2024 -> Jan 2025"], color="darkred", linestyle="--",
                    label="Official inflation 2024")
    axes[1].axvline(growth_2024.median(), color="green", linestyle="--", label="Median growth 2024 vs 2023")
    axes[1].axvline(growth_2025.median(), color="darkgreen", linestyle="--", label="Median growth 2025 vs 2024")
    axes[1].set_title("Same-Store January RTO Growth, Middle 90% of Stores")
    axes[1].set_xlabel("Growth rate")
    axes[1].set_ylabel("Density")
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    axes[1].legend()

    plt.tight_layout()
    st.pyplot(plt.gcf())

    st.markdown("""

    To avoid overclaiming, we compare turnover growth with operational features. If turnover grows much faster than traffic, working hours, cash registers, and the number of items in a receipt, then growth in the monetary value of purchases becomes a more plausible explanation.

    """)

    jan_features = clean[(clean["Месяц"] == 1) & (clean["Год"].isin([2023, 2024, 2025]))].copy()

    features_to_compare = [
        "РТО",
        "Среднее количество товаров в чеке",
        "Среднее количество промо товаров ",
        "promo_share",
        "total_traffic",
        "Количество касс",
        "Рабочие часы в день",
    ]

    jan_pivot = jan_features.pivot(index="new_id", columns="Год", values=features_to_compare).dropna()

    growth_rows = []
    for feature in features_to_compare:
        growth_2024_feature = jan_pivot[(feature, 2024)] / jan_pivot[(feature, 2023)] - 1
        growth_2025_feature = jan_pivot[(feature, 2025)] / jan_pivot[(feature, 2024)] - 1

        growth_rows.append({
            "feature": feature,
            "period": "Jan 2023 -> Jan 2024",
            "median_growth": growth_2024_feature.median(),
            "mean_growth": growth_2024_feature.mean(),
            "share_positive_growth": (growth_2024_feature > 0).mean(),
        })
        growth_rows.append({
            "feature": feature,
            "period": "Jan 2024 -> Jan 2025",
            "median_growth": growth_2025_feature.median(),
            "mean_growth": growth_2025_feature.mean(),
            "share_positive_growth": (growth_2025_feature > 0).mean(),
        })

    growth_comparison = pd.DataFrame(growth_rows)
    st.markdown("### How turnover and other features grew")
    st.dataframe(growth_comparison)

    plt.figure(figsize=(14, 7))
    sns.barplot(data=growth_comparison, x="median_growth", y="feature", hue="period")
    plt.axvline(0, color="black", linewidth=1)
    plt.axvline(official_inflation["Jan 2023 -> Jan 2024"], color="red", linestyle="--",
                label="Official inflation 2023")
    plt.axvline(official_inflation["Jan 2024 -> Jan 2025"], color="darkred", linestyle="--",
                label="Official inflation 2024")
    plt.title("Median Same-Store Growth of Turnover and Related Features")
    plt.xlabel("Median growth rate")
    plt.ylabel("Feature")
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt.gcf())

    rto_growth = growth_comparison[growth_comparison["feature"] == "РТО"][["period", "median_growth"]]
    other_growth = growth_comparison[growth_comparison["feature"] != "РТО"][["period", "feature", "median_growth"]]

    rto_vs_features = other_growth.merge(rto_growth, on="period", suffixes=("_feature", "_RTO"))
    rto_vs_features["RTO_growth_minus_feature_growth"] = (
            rto_vs_features["median_growth_RTO"] - rto_vs_features["median_growth_feature"]
    )

    st.markdown("### Turnover growth minus feature growth")
    st.dataframe(rto_vs_features.sort_values(["period", "RTO_growth_minus_feature_growth"], ascending=[True, False]))

    st.markdown("""
    **Hypothesis:** larger stores generate more turnover per cash register — and this effect holds *within both* alcohol license groups.

    This is a stronger check than simply comparing store sizes. Stores with an alcohol license tend to have a higher average receipt value, which could inflate their turnover per cash register. If larger stores also happen to have alcohol licenses more often, the area effect might just be a proxy for that. The hypothesis checks whether the area premium survives when the license group is held constant.
    """)

    area_efficiency = (
        clean.groupby("Торговая площадь, категориальный")
        .agg(
            observations=("new_id", "size"),
            stores=("new_id", "nunique"),
            median_RTO=("РТО", "median"),
            mean_RTO=("РТО", "mean"),
            median_cash_registers=("Количество касс", "median"),
            median_RTO_per_cash_register=("rto_per_cash_register", "median"),
            mean_RTO_per_cash_register=("rto_per_cash_register", "mean"),
            median_total_traffic=("total_traffic", "median"),
        )
        .reset_index()
    )

    area_order = area_efficiency.sort_values("median_RTO")["Торговая площадь, категориальный"]

    st.markdown("### Turnover per cash register by store size")
    st.dataframe(area_efficiency[[
        "Торговая площадь, категориальный",
        "stores",
        "median_RTO",
        "median_cash_registers",
        "median_RTO_per_cash_register",
        "median_total_traffic",
    ]].sort_values("median_RTO"))

    area_order_fixed = (
        clean.groupby("Торговая площадь, категориальный")["rto_per_cash_register"]
        .median()
        .sort_values()
        .index
    )

    plt.figure(figsize=(14, 6))
    sns.boxplot(
        data=clean,
        x="Торговая площадь, категориальный",
        y="rto_per_cash_register",
        hue="Флаг алкогольной лицензии",
        order=area_order_fixed,
    )
    plt.title("Turnover per Cash Register by Store Area and Alcohol License")
    plt.xlabel("Store area category")
    plt.ylabel("РТО per cash register")
    plt.legend(title="Alcohol license")
    plt.tight_layout()
    st.pyplot(plt.gcf())

    h2_table = (
        clean
        .groupby(["Торговая площадь, категориальный", "Флаг алкогольной лицензии"])["rto_per_cash_register"]
        .median()
        .unstack("Флаг алкогольной лицензии")
        .loc[area_order_fixed]
    )
    h2_table.columns = ["No license (0)", "Has license (1)"]
    h2_table.index.name = "Store area"

    smallest = area_order_fixed[0]
    largest = area_order_fixed[-1]

    h2_table.loc["Delta (largest − smallest)"] = (
            h2_table.loc[largest] - h2_table.loc[smallest]
    )

    st.markdown("### Median RTO per cash register: store size × alcohol license")
    st.dataframe(h2_table.round(0))

    st.markdown("### Statistical test (Mann-Whitney): bigger stores vs smallest ones")
    for license_flag in [0, 1]:
        group = clean[clean["Флаг алкогольной лицензии"] == license_flag]
        small = group[group["Торговая площадь, категориальный"] == smallest]["rto_per_cash_register"].dropna()
        large = group[group["Торговая площадь, категориальный"] == largest]["rto_per_cash_register"].dropna()
        stat, p = mannwhitneyu(large, small, alternative="greater")
        label = "No license" if license_flag == 0 else "Has license"
        st.write(f"{label}: Mann-Whitney U = {stat:.0f}, p = {p:.2e}")

elif window == "Conclusion":
    if "clean" not in st.session_state:
        st.error("Cleanup")
        st.stop()
    clean = st.session_state.clean

    st.markdown("""
    This project analyzed a monthly store-level dataset for Pyaterochka stores. The dataset contains store identifiers, calendar fields, turnover, store characteristics, local traffic indicators, regional information, and categorical descriptions of store area and age.

    The first important finding is that a dataset can have no explicit missing values and still require serious cleanup. Several hidden data quality problems were found: impossible working hours, suspiciously low working hours, zero population values, zero traffic values, and zero cash registers. These values were treated as hidden missing or incorrect values, replaced with `NaN`, and then filled using group medians. At the same time, anomaly flags were kept to preserve information about the original data problems.

    Descriptive statistics and visualizations showed that turnover is strongly right-skewed. Most stores have moderate turnover, while a smaller number of stores have very high turnover. The general overview also showed stable store coverage across months, seasonal turnover patterns, differences between store area categories, and skewed traffic distributions.

    The detailed overview showed that regional total turnover is strongly affected by the number of stores in the dataset, so median turnover and store-level comparisons are more useful for comparing typical performance. Correlation and scatter plots showed that turnover is associated with store capacity indicators such as cash registers and with receipt-related variables.

    The first hypothesis was partially supported: same-store January turnover growth was higher than the official inflation benchmark in both tested periods. However, this result cannot be interpreted as direct proof of food price inflation, because the dataset does not contain product-level prices, number of receipts, or average receipt value in rubles. The second hypothesis was supported in both alcohol license groups: larger stores have higher turnover per cash register regardless of whether they sell alcohol, which confirms that the area effect is not a proxy for the license.

    The main limitations are the absence of product-level prices, exact store area in square meters, number of receipts, and official regional store counts. Therefore, the analysis should be interpreted as exploratory, not causal. The web interface is developed separately and presents the final report, plots, explanations, and analytical outputs in an interactive format.

    """)
