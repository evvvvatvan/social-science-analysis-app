import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
from scipy import stats
import matplotlib.pyplot as plt

st.set_page_config(page_title="Data Analysis App", layout="wide")
st.title("Data Analysis App")

# =========================================================
# Helpers
# =========================================================
def load_data():
    st.header("1) Load data")

    source = st.radio(
        "Choose data source",
        ["Upload CSV", "Local CSV path"],
        horizontal=True,
        key="data_source"
    )

    df = None

    if source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], key="csv_upload")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("CSV loaded successfully.")
            except Exception as e:
                st.error(f"Could not read file: {e}")
    else:
        csv_path = st.text_input("Enter full CSV path", value="", key="csv_path")
        if csv_path:
            try:
                df = pd.read_csv(csv_path)
                st.success("CSV loaded successfully.")
            except Exception as e:
                st.error(f"Could not read file: {e}")

    return df


def safe_numeric_columns(df):
    return df.select_dtypes(include=np.number).columns.tolist()


def build_formula(y_var, x_vars, add_fe=False, fe_var=None):
    if not x_vars:
        return None

    formula = f"Q('{y_var}') ~ " + " + ".join([f"Q('{x}')" for x in x_vars])

    if add_fe and fe_var:
        formula += f" + C(Q('{fe_var}'))"

    return formula


def run_model_formula(data, formula, robust=False):
    model = smf.ols(formula=formula, data=data).fit()
    if robust:
        model = model.get_robustcov_results(cov_type="HC1")
    return model


def make_pretty_summary(models, names, robust=False):
    summary = summary_col(
        models,
        stars=True,
        model_names=names,
        info_dict={
            "N": lambda x: f"{int(x.nobs)}",
            "R2": lambda x: f"{x.rsquared:.3f}",
            "Adj. R2": lambda x: f"{x.rsquared_adj:.3f}",
            "SE Type": lambda x: "HC1" if robust else "OLS",
        },
    )
    return summary


def significance_stars(p):
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.1:
        return "*"
    return ""


def compute_cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return np.nan

    s1 = g1.std(ddof=1)
    s2 = g2.std(ddof=1)

    pooled_var = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)

    if pd.isna(pooled_var) or pooled_var <= 0:
        return np.nan

    pooled_sd = np.sqrt(pooled_var)
    return (g1.mean() - g2.mean()) / pooled_sd


def two_group_diff_table(df, group_col, vars_to_compare, group_a_label, group_b_label):
    results = []

    for var in vars_to_compare:
        temp = df[[group_col, var]].copy().dropna()

        g1 = temp.loc[temp[group_col] == group_a_label, var]
        g2 = temp.loc[temp[group_col] == group_b_label, var]

        if len(g1) == 0 or len(g2) == 0:
            continue

        try:
            t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False, nan_policy="omit")
        except Exception:
            t_stat, p_val = np.nan, np.nan

        mean_g1 = g1.mean()
        mean_g2 = g2.mean()
        mean_diff = mean_g1 - mean_g2
        cohen_d = compute_cohens_d(g1, g2)

        results.append({
            "variable": var,
            f"mean_{group_a_label}": mean_g1,
            f"mean_{group_b_label}": mean_g2,
            "mean_diff": mean_diff,
            "cohen_d": cohen_d,
            f"n_{group_a_label}": len(g1),
            f"n_{group_b_label}": len(g2),
            "t_stat": t_stat,
            "p_value": p_val,
        })

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results)
    out["sig"] = out["p_value"].apply(significance_stars)
    out["abs_cohen_d"] = out["cohen_d"].abs()
    out = out.sort_values("abs_cohen_d", ascending=False).reset_index(drop=True)
    return out


def format_p_value(p):
    if pd.isna(p):
        return np.nan
    if p < 0.0001:
        return "<0.0001"
    return f"{p:.4f}"


def format_group_comparison_table(df, group_a_label, group_b_label, show_full=False):
    if df.empty:
        return df

    out = df.copy()

    numeric_cols_4 = [
        f"mean_{group_a_label}",
        f"mean_{group_b_label}",
        "mean_diff",
        "cohen_d",
        "t_stat",
    ]
    for col in numeric_cols_4:
        if col in out.columns:
            out[col] = out[col].round(4)

    out["p_value"] = out["p_value"].apply(format_p_value)

    default_cols = [
        "variable",
        f"mean_{group_a_label}",
        f"mean_{group_b_label}",
        "mean_diff",
        "cohen_d",
        "p_value",
        "sig",
    ]

    full_cols = [
        "variable",
        f"mean_{group_a_label}",
        f"mean_{group_b_label}",
        "mean_diff",
        "cohen_d",
        f"n_{group_a_label}",
        f"n_{group_b_label}",
        "t_stat",
        "p_value",
        "sig",
    ]

    keep_cols = full_cols if show_full else default_cols
    keep_cols = [c for c in keep_cols if c in out.columns]

    return out[keep_cols]


def make_scatterplot(df, x_var, y_var, add_trendline=True, equal_axes=False):
    plot_df = df[[x_var, y_var]].dropna().copy()

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(plot_df[x_var], plot_df[y_var], alpha=0.6)

    if add_trendline and len(plot_df) >= 2:
        z = np.polyfit(plot_df[x_var], plot_df[y_var], 1)
        p = np.poly1d(z)
        x_line = np.linspace(plot_df[x_var].min(), plot_df[x_var].max(), 100)
        ax.plot(x_line, p(x_line))

    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    ax.set_title(f"{y_var} vs {x_var}")

    if equal_axes:
        all_min = min(plot_df[x_var].min(), plot_df[y_var].min())
        all_max = max(plot_df[x_var].max(), plot_df[y_var].max())
        ax.set_xlim(all_min, all_max)
        ax.set_ylim(all_min, all_max)
        ax.set_aspect("equal", adjustable="box")
        ax.plot([all_min, all_max], [all_min, all_max], linestyle="--")

    plt.tight_layout()
    return fig


# =========================================================
# Load once for all tabs
# =========================================================
df = load_data()

if df is None:
    st.stop()

st.write("Preview:")
st.dataframe(df.head())

st.write(f"Rows: {len(df):,}")
st.write(f"Columns: {len(df.columns)}")

all_columns = df.columns.tolist()
numeric_columns = safe_numeric_columns(df)

if len(all_columns) == 0:
    st.error("No columns found.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Regression", "Group Comparison", "Correlation"])

# =========================================================
# TAB 1: REGRESSION
# =========================================================
with tab1:
    st.header("Regression")

    if not numeric_columns:
        st.error("No numeric columns found in this dataset.")
    else:
        st.subheader("Dependent variable")
        y_var = st.selectbox("Select Y", options=numeric_columns, key="reg_y")

        st.subheader("Model options")
        add_fe = st.checkbox("Add fixed effects", value=False, key="add_fe")
        fe_var = None
        if add_fe:
            fe_candidates = [c for c in all_columns if c != y_var]
            fe_var = st.selectbox("Select fixed effect variable", options=fe_candidates, key="fe_var")

        robust = st.checkbox("Use robust standard errors (HC1)", value=True, key="robust_se")

        st.subheader("Build models")
        num_models = st.number_input(
            "Number of models",
            min_value=1,
            max_value=20,
            value=3,
            step=1,
            key="num_models"
        )

        model_specs = []
        model_names = []

        for i in range(int(num_models)):
            st.markdown(f"**Model {i+1}**")
            model_name = st.text_input(
                f"Model {i+1} name",
                value=f"Model {i+1}",
                key=f"model_name_{i}"
            )

            x_vars = st.multiselect(
                f"Select X variables for Model {i+1}",
                options=[c for c in all_columns if c != y_var and c != fe_var],
                default=[],
                key=f"xvars_{i}"
            )

            model_names.append(model_name)
            model_specs.append(x_vars)

        if st.button("Run Regression", key="run_regression"):
            fitted_models = []
            used_names = []

            for i, x_vars in enumerate(model_specs):
                if len(x_vars) == 0:
                    continue

                try:
                    formula = build_formula(
                        y_var=y_var,
                        x_vars=x_vars,
                        add_fe=add_fe,
                        fe_var=fe_var
                    )

                    model = run_model_formula(df, formula, robust=robust)
                    fitted_models.append(model)
                    used_names.append(model_names[i])

                except Exception as e:
                    st.error(f"Model {i+1} failed: {e}")

            if not fitted_models:
                st.warning("No valid models were run.")
            else:
                summary = make_pretty_summary(fitted_models, used_names, robust=robust)

                st.subheader("Regression summary")
                st.write(f"Standard errors: {'Robust (HC1)' if robust else 'Default OLS'}")

                html_table = summary.as_html()

                st.markdown("""
                <style>
                table {
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 14px;
                }
                th, td {
                    padding: 6px 10px;
                    text-align: right;
                    vertical-align: top;
                    white-space: nowrap;
                }
                th:first-child, td:first-child {
                    text-align: left;
                    white-space: normal;
                    min-width: 260px;
                }
                </style>
                """, unsafe_allow_html=True)

                st.markdown(html_table, unsafe_allow_html=True)

                st.download_button(
                    label="Download regression table (.html)",
                    data=html_table,
                    file_name="regression_results.html",
                    mime="text/html",
                    key="download_reg_html"
                )

                with st.expander("Show model formulas"):
                    for name, x_vars in zip(model_names, model_specs):
                        if x_vars:
                            formula = build_formula(
                                y_var=y_var,
                                x_vars=x_vars,
                                add_fe=add_fe,
                                fe_var=fe_var
                            )
                            st.code(f"{name}: {formula}")

# =========================================================
# TAB 2: GROUP COMPARISON
# =========================================================
with tab2:
    st.header("Group Comparison / Difference in Means")

    if not numeric_columns:
        st.error("No numeric columns found in this dataset.")
    else:
        compare_mode = st.radio(
            "Choose grouping method",
            ["Existing binary/categorical variable", "Threshold rule", "Top N vs Bottom N"],
            horizontal=True,
            key="compare_mode"
        )

        comp_df = df.copy()
        temp_group_col = "__group__"
        group_a = None
        group_b = None

        if compare_mode == "Existing binary/categorical variable":
            group_var = st.selectbox("Select grouping variable", options=all_columns, key="group_var_existing")

            non_null_values = sorted(comp_df[group_var].dropna().astype(str).unique().tolist())
            if len(non_null_values) < 2:
                st.warning("Selected grouping variable does not have at least two non-missing groups.")
            else:
                group_a = st.selectbox("Choose Group A", options=non_null_values, key="group_a_existing")
                group_b_candidates = [x for x in non_null_values if x != group_a]
                group_b = st.selectbox("Choose Group B", options=group_b_candidates, key="group_b_existing")
                comp_df[temp_group_col] = comp_df[group_var].astype(str)

        elif compare_mode == "Threshold rule":
            threshold_var = st.selectbox("Select threshold variable", options=numeric_columns, key="threshold_var")
            threshold = st.number_input("Threshold value", value=0.0, key="threshold_value")
            threshold_rule = st.radio(
                "Group A rule",
                ["Greater than or equal to threshold", "Greater than threshold"],
                horizontal=True,
                key="threshold_rule"
            )

            group_a = "GroupA"
            group_b = "GroupB"

            if threshold_rule == "Greater than or equal to threshold":
                comp_df[temp_group_col] = np.where(comp_df[threshold_var] >= threshold, group_a, group_b)
            else:
                comp_df[temp_group_col] = np.where(comp_df[threshold_var] > threshold, group_a, group_b)

        else:
            rank_var = st.selectbox("Select ranking variable", options=numeric_columns, key="rank_var")
            n_each = st.number_input("Number in top and bottom groups", min_value=1, value=25, step=1, key="n_each")

            temp = comp_df[[rank_var]].copy().dropna().sort_values(rank_var)
            if len(temp) < 2 * n_each:
                st.warning("Not enough non-missing observations for top/bottom split.")
            else:
                bottom_idx = temp.head(n_each).index
                top_idx = temp.tail(n_each).index

                comp_df[temp_group_col] = np.nan
                comp_df.loc[bottom_idx, temp_group_col] = "Bottom"
                comp_df.loc[top_idx, temp_group_col] = "Top"

                group_a = "Top"
                group_b = "Bottom"

        vars_to_compare = st.multiselect(
            "Select variables to compare",
            options=numeric_columns,
            default=[],
            key="vars_to_compare"
        )

        show_full_stats = st.checkbox("Show full statistics", value=False, key="show_full_diff")

        if st.button("Run Group Comparison", key="run_group_comparison"):
            if not vars_to_compare:
                st.warning("Please choose at least one variable to compare.")
            elif temp_group_col not in comp_df.columns or group_a is None or group_b is None:
                st.warning("Grouping rule is not ready.")
            else:
                valid_df = comp_df[[temp_group_col] + vars_to_compare].copy()
                valid_df = valid_df.dropna(subset=[temp_group_col])

                result = two_group_diff_table(
                    df=valid_df,
                    group_col=temp_group_col,
                    vars_to_compare=vars_to_compare,
                    group_a_label=group_a,
                    group_b_label=group_b
                )

                display_result = format_group_comparison_table(
                    result,
                    group_a_label=group_a,
                    group_b_label=group_b,
                    show_full=show_full_stats
                )

                if display_result.empty:
                    st.warning("No valid comparison results.")
                else:
                    st.subheader("Difference-in-means table")
                    st.dataframe(display_result, use_container_width=True)

                    csv_data = display_result.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download difference-in-means table (.csv)",
                        data=csv_data,
                        file_name="difference_in_means.csv",
                        mime="text/csv",
                        key="download_diff_csv"
                    )

# =========================================================
# TAB 3: CORRELATION
# =========================================================
with tab3:
    st.header("Correlation Explorer")

    if len(numeric_columns) < 2:
        st.error("Need at least two numeric columns.")
    else:
        x_var = st.selectbox("Select X variable", options=numeric_columns, key="corr_x")
        y_candidates = [c for c in numeric_columns if c != x_var]
        y_var_corr = st.selectbox("Select Y variable", options=y_candidates, key="corr_y")

        add_trendline = st.checkbox("Add trendline", value=True, key="corr_trendline")
        equal_axes = st.checkbox("Use equal axes + 45-degree line", value=False, key="corr_equal_axes")

        if st.button("Run Correlation", key="run_correlation"):
            corr_df = df[[x_var, y_var_corr]].dropna().copy()

            if len(corr_df) < 3:
                st.warning("Not enough non-missing observations.")
            else:
                pearson_r, pearson_p = stats.pearsonr(corr_df[x_var], corr_df[y_var_corr])
                spearman_rho, spearman_p = stats.spearmanr(corr_df[x_var], corr_df[y_var_corr])

                corr_results = pd.DataFrame({
                    "metric": ["Pearson r", "Pearson p", "Spearman rho", "Spearman p", "N"],
                    "value": [
                        round(pearson_r, 4),
                        format_p_value(pearson_p),
                        round(spearman_rho, 4),
                        format_p_value(spearman_p),
                        len(corr_df),
                    ]
                })

                st.subheader("Correlation results")

                col1, col2 = st.columns([1, 1])

                with col1:
                    st.dataframe(corr_results, use_container_width=True)

                with col2:
                    fig = make_scatterplot(
                        df=corr_df,
                        x_var=x_var,
                        y_var=y_var_corr,
                        add_trendline=add_trendline,
                        equal_axes=equal_axes
                    )
                    st.pyplot(fig)

                csv_data = corr_results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download correlation results (.csv)",
                    data=csv_data,
                    file_name="correlation_results.csv",
                    mime="text/csv",
                    key="download_corr_csv"
                )