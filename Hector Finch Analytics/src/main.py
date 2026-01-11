import os
import re
import signal
import warnings
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
import json
import difflib

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display

from data_processing import (
    load_and_prepare_data,
    process_data,
    remove_upper_outliers_quantile_isolation_forest,
    create_grouped_dataframes,
    compute_statistics,
    project_with_regression_yearly,
    apply_attribute_whitelists,
    run_transformation,
    filter_by_codes_and_ranges,
    validate_core_code_datasets
)

# Ignore future warnings (if any)
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Configuration & Utility Functions ---
def load_mappings_file(path: Path) -> pd.DataFrame:
    """
    Load the 'Unique Product Code Values' mapping table and normalise headers.

    Canonicalises to:
        Original  -> raw product code / specifier token
        Mapped    -> cleaned/normalised version (will use 'Stored As' if present)

    Falls back to similarly named CSV in the same folder if the expected file
    is missing. Tolerates BOMs, whitespace, and case differences in headers.
    """
    if not path.exists():
        # fallback to any similarly named CSV in the same folder
        alts = list(path.parent.glob(f"*{path.stem}*.csv"))
        if alts:
            print(f"[WARN] '{path.name}' not found—using '{alts[0].name}' instead.")
            path = alts[0]
        else:
            raise FileNotFoundError(f"Could not find mappings file: {path}")

    # try default encoding, then latin-1
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")

    # strip whitespace/BOM from headers
    df.rename(columns=lambda c: c.strip().lstrip("\ufeff"), inplace=True)
    lower_map = {c.lower(): c for c in df.columns}

    # candidate headers for the source (raw/original) values
    orig_candidates = (
        "original",
        "original product code",
        "original code",
        "orig",
        "raw",
        "before",
    )

    # candidate headers for the mapped/cleaned value
    # IMPORTANT: your file shows ['Original', 'Category', 'Stored As'], so we include 'stored as'
    mapped_candidates = (
        "mapped",
        "mapped value",
        "cleaned",
        "after",
        "replacement",
        "standard",
        "stored as",
        "stored_as",
        "storedas",
    )

    def _resolve(candidates):
        for cand in candidates:
            if cand in lower_map:
                return lower_map[cand]
        return None

    orig = _resolve(orig_candidates)
    mapped = _resolve(mapped_candidates)

    if orig is None:
        raise KeyError(
            f"Could not locate an 'Original' column in {path.name}. Found columns: {list(df.columns)}"
        )
    if mapped is None:
        # final fallback: if the file has only 2 or 3 cols and one is 'Category',
        # we assume the last non-Original column is the mapped.
        fallback_cols = [c for c in df.columns if c != lower_map[orig.lower()]]
        # prefer 'Stored As' if present, else last remaining
        if any(c.lower() == "stored as" for c in fallback_cols):
            mapped = next(c for c in fallback_cols if c.lower() == "stored as")
        elif fallback_cols:
            mapped = fallback_cols[-1]
            print(f"[WARN] No mapped column found; falling back to '{mapped}' in {path.name}.")
        else:
            raise KeyError(
                f"Could not locate a mapped/cleaned column in {path.name}. Found columns: {list(df.columns)}"
            )

    # rename canonical
    df.rename(columns={orig: 'Original', mapped: 'Mapped'}, inplace=True)

    print(f"[INFO] Mappings: using '{orig}' → 'Original', '{mapped}' → 'Mapped' from {path.name}.")
    return df

try:
    script_dir = Path(__file__).parent.resolve()
except NameError:
    script_dir = Path().resolve()

project_root = script_dir.parent
data_dir = project_root / 'data' / 'Program Data'
figure_dir = project_root / 'figures'
CONFIG_FILE = project_root / "config.json"

def get_default_params():
    """Returns a dictionary of the script's default parameters."""
    return {
        "start_date": None,
        "end_date": None,
        "default_start": None,
        "product_codes": None,
        "contamination": 0.01,
        "max_trendlines": 3,
        "show_labels_yearly": True,
        "show_labels_quarterly": True,
        "show_labels_monthly": False,
        "max_lines_for_labels": 3,
        "max_lines_for_legend": 20,
        "other_threshold": 2.5,
        "show_info_box_yearly": True,
        "show_info_box_quarterly": True,
        "show_info_box_monthly": True,
        "max_lines_for_info_box": 3,
        "enable_advanced_filtering": False,
        "save_figures": True,
        "skip_transformation": False,
        "download_data": False,
        "enable_plotting": True,
    }

def save_params(params: dict, profile_name: str):
    """Saves the given parameters to a named profile in the config file."""
    serializable_params = params.copy()
    for key, value in serializable_params.items():
        if isinstance(value, datetime):
            serializable_params[key] = value.isoformat()
        elif isinstance(value, Path):
            serializable_params[key] = str(value)

    config_data = {"profiles": {}}
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                config_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            print(f"Warning: Could not read {CONFIG_FILE}. A new config will be created.")
            pass

    if "profiles" not in config_data or not isinstance(config_data["profiles"], dict):
        config_data["profiles"] = {}
        
    config_data["profiles"][profile_name] = serializable_params
    config_data["__last_used__"] = profile_name

    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config_data, f, indent=4)
        print(f"Parameters saved to profile '{profile_name}' in {CONFIG_FILE}")
    except Exception as e:
        print(f"Error saving parameters: {e}")

def load_params(default_params: dict, profile_name: str = None) -> dict:
    """
    Loads parameters from the config file.
    - If profile_name is given, loads that profile.
    - If not, loads the last used profile.
    - Merges with default_params to ensure all keys are present.
    """
    if not CONFIG_FILE.exists():
        return default_params

    try:
        with open(CONFIG_FILE, 'r') as f:
            config_data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading config file: {e}. Using default parameters.")
        return default_params

    profiles = config_data.get("profiles", {})
    if not profile_name:
        profile_name = config_data.get("__last_used__", "default")

    loaded_params = profiles.get(profile_name, {})
    if loaded_params:
        print(f"Parameters loaded from profile: '{profile_name}'")
    else:
        if profile_name != "default":
             print(f"Profile '{profile_name}' not found. Using default parameters.")
        return default_params

    # Deserialize date strings back to datetime objects
    for key, value in loaded_params.items():
        if key in ['start_date', 'end_date', 'earliest_date', 'latest_date', 'default_start'] and isinstance(value, str):
            try:
                loaded_params[key] = datetime.fromisoformat(value)
            except (ValueError, TypeError):
                pass
    
    # Merge loaded params into defaults to ensure all keys exist
    final_params = default_params.copy()
    final_params.update(loaded_params)
    return final_params

def create_new_figure_directory(base_path: Path, base_name: str, save_figures: bool, download_data: bool = False) -> Path:
    today_date = datetime.today().strftime('%d-%m-%Y')
    i = 1
    new_dir = base_path / f"{base_name} - Breakdown ({today_date}) - {i}"
    while new_dir.exists():
        i += 1
        new_dir = base_path / f"{base_name} - Breakdown ({today_date}) - {i}"
    if save_figures or download_data:
        new_dir.mkdir(parents=True, exist_ok=True)
    return new_dir

# --- Chart Creation & Data Download Functions ---

def create_charts(
    grouped_dfs: dict,
    stats_results: dict,
    figure_dir: Path,
    save_figures: bool,
    chosen_categories_str: str = "",
    max_trendlines: int = 3,
    merge_small_sections: bool = True,
    other_threshold: float = 2.5,
    show_labels_yearly: bool = True,
    show_labels_quarterly: bool = True,
    show_labels_monthly: bool = False,
    max_lines_for_labels: int = 3,
    max_lines_for_legend: int = 20,
    show_info_box_yearly: bool = True,
    show_info_box_quarterly: bool = True,
    show_info_box_monthly: bool = True,
    max_lines_for_info_box: int = 3,
    download_data: bool = False,
    enable_plotting: bool = True,
    valid_items: pd.DataFrame = None,
    color_maps: dict = None,
):
    sns.set_theme(style="whitegrid")
    available_graphs = []
    overall_data = grouped_dfs.get("overall", {})

    # --- Helper functions (nested) ---
    def sanitize_filename(name: str) -> str:
        return re.sub(r'[^\w\-_\. ]', '_', name)

    def place_summary_text(ax, summary_lines):
        ax.text(
            0.01, 0.99,
            "\n".join(summary_lines),
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8),
        )

    def compute_period_changes(df, period_type: str):
        if period_type == 'year':
            df = df.sort_values('Year')
        elif period_type == 'quarter':
            df = df.sort_values('Quarter_Num_Total')
        else:
            df = df.sort_values('Month.Year.Date')
        df['Pct_Change'] = df['Adjusted Net Value'].pct_change() * 100
        df['Pct_Change'].replace([float("inf"), float("-inf")], float("nan"), inplace=True)
        return df

    def should_show_labels(period_type: str, line_count: int) -> bool:
        if line_count > max_lines_for_labels:
            return False
        if period_type == 'year':
            return show_labels_yearly
        elif period_type == 'quarter':
            return show_labels_quarterly
        return show_labels_monthly

    def should_show_legend(line_count: int) -> bool:
        return line_count <= max_lines_for_legend

    def should_place_info_box(period_type: str, line_count: int) -> bool:
        if line_count > max_lines_for_info_box:
            return False
        if period_type == 'year':
            return show_info_box_yearly
        elif period_type == 'quarter':
            return show_info_box_quarterly
        elif period_type == 'month':
            return show_info_box_monthly
        return False

    # --- Overall Yearly Sales ---
    def plot_overall_yearly_sales():
        data = overall_data.get("year", pd.DataFrame()).copy()
        if data.empty:
            print("No yearly data for Overall.")
            return

        data = compute_period_changes(data, 'year')
        avg_yoy = data['Pct_Change'].mean()

        fig, ax = plt.subplots(figsize=(12, 7))
        sns.lineplot(data=data, x="Year", y="Adjusted Net Value", marker="o",
                     color="teal", linewidth=2.5, label="Overall Sales", ax=ax)

        if len(data) > 1 and max_trendlines >= 1:
            z = np.polyfit(data["Year"], data["Adjusted Net Value"], 1)
            p = np.poly1d(z)
            ax.plot(data["Year"], p(data["Year"]), "r--", label="Trend Line")

        if should_show_labels('year', 1):
            for i in range(1, len(data)):
                if pd.notnull(data["Pct_Change"].iloc[i]):
                    ax.text(data["Year"].iloc[i], data["Adjusted Net Value"].iloc[i],
                            f"{data['Pct_Change'].iloc[i]:.1f}%", ha="center", va="bottom", fontsize=9, color="red")

        ax.set_title(f"Overall Yearly Sales Trend (Cats: {chosen_categories_str})", fontsize=16)
        ax.set_xlabel("Year", fontsize=14)
        ax.set_ylabel("Adjusted Net Value", fontsize=14)
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
        ax.tick_params(axis="both", which="major", labelsize=12)
        if should_show_legend(1):
            ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=12)
        else:
            ax.get_legend().remove()
        plt.tight_layout()
        if should_place_info_box('year', 1) and not np.isnan(avg_yoy):
            place_summary_text(ax, [f"Average YoY Growth: {avg_yoy:.2f}%"])
        if save_figures:
            plt.savefig(figure_dir / "Overall_Yearly_Sales_Trend.png", bbox_inches="tight")
        
        if download_data:
            chart_data = data.copy().sort_values(by='Year', ascending=False)
            fname = sanitize_filename("Overall_Yearly_Sales_Trend_Data.csv")
            figure_dir.mkdir(parents=True, exist_ok=True)
            chart_data.to_csv(figure_dir / fname, index=False)
            print(f"Raw data saved to {fname}")

    available_graphs.append(("Overall Yearly Sales Trend", plot_overall_yearly_sales))

    # --- Overall Quarterly Sales ---
    def plot_overall_quarterly_sales():
        data = overall_data.get("quarter", pd.DataFrame()).copy()
        if data.empty:
            print("No quarterly data for Overall.")
            return
        data["Quarter"] = data["Quarter"].astype(str)
        data["Year"] = data["Quarter"].str[:4].astype(int)
        data["Quarter_Num"] = data["Quarter"].str[-1].astype(int)
        data["Quarter_Num_Total"] = data["Year"] * 4 + data["Quarter_Num"] - 1
        data = compute_period_changes(data, 'quarter')
        avg_qoq = data['Pct_Change'].mean()

        fig, ax = plt.subplots(figsize=(12, 7))
        sns.lineplot(data=data, x="Quarter_Num_Total", y="Adjusted Net Value",
                     marker="o", color="purple", linewidth=2.5, label="Overall Sales", ax=ax)
        ax.set_xticks(data["Quarter_Num_Total"])
        ax.set_xticklabels(data["Quarter"], rotation=45)
        if len(data) > 1 and max_trendlines >= 1:
            z = np.polyfit(data["Quarter_Num_Total"], data["Adjusted Net Value"], 1)
            p = np.poly1d(z)
            ax.plot(data["Quarter_Num_Total"], p(data["Quarter_Num_Total"]), "r--", label="Trend Line")
        for yr in sorted(data["Year"].unique())[:-1]:
            last_q = data.loc[data["Year"] == yr, "Quarter_Num_Total"].max()
            ax.axvline(x=last_q + 0.5, color="gray", linestyle="--", alpha=0.7)
        if should_show_labels('quarter', 1):
            for i in range(1, len(data)):
                if pd.notnull(data["Pct_Change"].iloc[i]):
                    ax.text(data["Quarter_Num_Total"].iloc[i], data["Adjusted Net Value"].iloc[i],
                            f"{data['Pct_Change'].iloc[i]:.1f}%", ha="center", va="bottom", fontsize=9, color="red")
        ax.set_title(f"Overall Quarterly Sales Trend (Cats: {chosen_categories_str})", fontsize=16)
        ax.set_xlabel("Quarter", fontsize=14)
        ax.set_ylabel("Adjusted Net Value", fontsize=14)
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
        ax.tick_params(axis="both", which="major", labelsize=12)
        if should_show_legend(1):
            ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=12)
        else:
            ax.get_legend().remove()
        plt.tight_layout()
        if should_place_info_box('quarter', 1) and not np.isnan(avg_qoq):
            place_summary_text(ax, [f"Average QoQ Growth: {avg_qoq:.2f}%"])
        if save_figures:
            plt.savefig(figure_dir / "Overall_Quarterly_Sales_Trend.png", bbox_inches="tight")

        if enable_plotting:
            plt.show()
        else:
            plt.close()

        if download_data:
            chart_data = data.copy().sort_values(by='Quarter_Num_Total', ascending=False)
            fname = sanitize_filename("Overall_Quarterly_Sales_Trend_Data.csv")
            figure_dir.mkdir(parents=True, exist_ok=True)
            chart_data.to_csv(figure_dir / fname, index=False)
            print(f"Raw data saved to {fname}")

    available_graphs.append(("Overall Quarterly Sales Trend", plot_overall_quarterly_sales))

    # --- Overall Monthly Sales ---
    def plot_overall_monthly_sales():
        data = overall_data.get("month", pd.DataFrame()).copy()
        if data.empty:
            print("No monthly data for Overall.")
            return
        data["Month.Year.Date"] = pd.to_datetime(data["Month.Year.Date"], errors='coerce')
        data.sort_values("Month.Year.Date", inplace=True)
        data['Pct_Change'] = data['Adjusted Net Value'].pct_change() * 100
        data['Pct_Change'].replace([float("inf"), float("-inf")], float("nan"), inplace=True)
        avg_mom = data['Pct_Change'].mean()

        fig, ax = plt.subplots(figsize=(14, 7))
        sns.lineplot(data=data, x="Month.Year.Date", y="Adjusted Net Value",
                     marker="o", color="darkgreen", linewidth=2.5, label="Overall Sales", ax=ax)
        if len(data) > 1 and max_trendlines >= 1:
            x_num = mdates.date2num(data["Month.Year.Date"])
            z = np.polyfit(x_num, data["Adjusted Net Value"], 1)
            p = np.poly1d(z)
            ax.plot(data["Month.Year.Date"], p(x_num), "r--", label="Trend Line")
        if should_show_labels('month', 1):
            for i in range(1, len(data)):
                if pd.notnull(data["Pct_Change"].iloc[i]):
                    ax.text(data["Month.Year.Date"].iloc[i], data["Adjusted Net Value"].iloc[i],
                            f"{data['Pct_Change'].iloc[i]:.1f}%", ha="center", va="bottom", fontsize=9, color="red")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        ax.set_title(f"Overall Monthly Sales Trend (Cats: {chosen_categories_str})", fontsize=16)
        ax.set_xlabel("Month", fontsize=14)
        ax.set_ylabel("Adjusted Net Value", fontsize=14)
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
        ax.tick_params(axis="both", which="major", labelsize=12)
        if should_show_legend(1):
            ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=12)
        else:
            ax.get_legend().remove()
        plt.tight_layout()
        if should_place_info_box('month', 1) and not np.isnan(avg_mom):
            place_summary_text(ax, [f"Average MoM Growth: {avg_mom:.2f}%"])
        if save_figures:
            plt.savefig(figure_dir / "Overall_Monthly_Sales_Trend.png", bbox_inches="tight")
        if enable_plotting:
            plt.show()
        else:
            plt.close()
        if download_data:
            chart_data = data.copy().sort_values(by='Month.Year.Date', ascending=False)
            fname = sanitize_filename("Overall_Monthly_Sales_Trend_Data.csv")
            figure_dir.mkdir(parents=True, exist_ok=True)
            chart_data.to_csv(figure_dir / fname, index=False)
            print(f"Raw data saved to {fname}")

    available_graphs.append(("Overall Monthly Sales Trend", plot_overall_monthly_sales))

    # --- Category-based Graphs ---
    for category, data_dict in grouped_dfs.items():
        if category == "overall" or category == "overall_quantity" or category.endswith("_quantity"):
            continue
        # Pie chart for overall distribution by attribute category
        if "overall" in data_dict and not data_dict["overall"].empty:
            def plot_pie_chart(category=category, overall_df=data_dict["overall"].copy()):
                if merge_small_sections:
                    total_value = overall_df["Adjusted Net Value"].sum()
                    if total_value > 0:
                        overall_df["Percentage"] = (overall_df["Adjusted Net Value"] / total_value) * 100
                    else:
                        overall_df["Percentage"] = 0
                    
                    if overall_df[overall_df["Percentage"] < other_threshold].shape[0] > 0:
                        overall_df[category] = overall_df.apply(
                            lambda row: "Other" if row["Percentage"] < other_threshold else row[category], axis=1
                        )
                        agg_spec = {'Adjusted Net Value': 'sum'}
                        if 'Quantity' in overall_df.columns:
                            agg_spec['Quantity'] = 'sum'
                        overall_df = overall_df.groupby(category, as_index=False).agg(agg_spec)

                fig, ax = plt.subplots(figsize=(8, 6))
                sum_val = overall_df["Adjusted Net Value"].sum()
                if sum_val == 0:
                    plt.title(f"No sales for {category}.")
                    if enable_plotting:
                        plt.show()
                    else:
                        plt.close()
                    return
                overall_df["Percentage"] = overall_df["Adjusted Net Value"] / sum_val * 100
                overall_df.sort_values("Adjusted Net Value", ascending=False, inplace=True)
                ax.pie(overall_df["Adjusted Net Value"],
                       labels=overall_df[category],
                       autopct="%1.1f%%", startangle=140)
                ax.set_title(f"Sales Distribution by {category}\n(Cats: {chosen_categories_str})", fontsize=16)
                plt.tight_layout()
                if save_figures:
                    fname = f"Sales_Distribution_by_{category}.png"
                    plt.savefig(figure_dir / fname, bbox_inches="tight")
                if enable_plotting:
                    plt.show()
                else:
                    plt.close()
                if download_data:
                    out_data = overall_df.copy().sort_values("Adjusted Net Value", ascending=False)
                    fname = sanitize_filename(f"Sales_Distribution_by_{category}_Data.csv")
                    figure_dir.mkdir(parents=True, exist_ok=True)
                    out_data.to_csv(figure_dir / fname, index=False)
                    print(f"Raw data for {category} distribution saved to {fname}")
            available_graphs.append((f"Sales Distribution by {category} (Overall)", plot_pie_chart))

        # Line charts by period for each category
        for period in ["year", "quarter", "month"]:
            if period in data_dict and not data_dict[period].empty:
                def plot_category_sales(period=period, period_df=data_dict[period].copy(), category=category):
                    if period == "quarter":
                        period_df["Quarter"] = period_df["Quarter"].astype(str)
                        period_df["Year"] = period_df["Quarter"].str[:4].astype(int)
                        period_df["Quarter_Num"] = period_df["Quarter"].str[-1].astype(int)
                        period_df["Quarter_Num_Total"] = period_df["Year"] * 4 + period_df["Quarter_Num"] - 1
                        x_col = "Quarter_Num_Total"
                        period_type = "quarter"
                        period_df.sort_values("Quarter_Num_Total", inplace=True)
                    elif period == "month":
                        period_df["Month.Year.Date"] = pd.to_datetime(period_df["Month.Year.Date"], errors='coerce')
                        x_col = "Month.Year.Date"
                        period_type = "month"
                        period_df.sort_values("Month.Year.Date", inplace=True)
                    else:
                        x_col = "Year"
                        period_type = "year"
                        period_df.sort_values("Year", inplace=True)

                    unique_cats = period_df[category].unique()
                    line_count = len(unique_cats)
                    fig, ax = plt.subplots(figsize=(14, 7) if period == "month" else (12, 7))
                    
                    palette_map = None
                    if color_maps and category in color_maps:
                        palette_map = color_maps[category]

                    for cat_val in unique_cats:
                        df_cat = period_df[period_df[category].isin([cat_val])].copy()
                        if df_cat.empty:
                            continue
                        
                        line_color = palette_map.get(cat_val) if palette_map else None
                        
                        df_cat['Pct_Change'] = df_cat['Adjusted Net Value'].pct_change() * 100
                        df_cat['Pct_Change'].replace([float("inf"), float("-inf")], np.nan, inplace=True)
                        
                        sns.lineplot(data=df_cat, x=x_col, y="Adjusted Net Value",
                                     marker="o", linewidth=2.0, label=cat_val, ax=ax, color=line_color)
                        
                        if line_count <= max_trendlines and len(df_cat) > 1:
                            numeric_x = mdates.date2num(df_cat[x_col]) if period_type == 'month' else df_cat[x_col]
                            z = np.polyfit(numeric_x, df_cat["Adjusted Net Value"], 1)
                            p = np.poly1d(z)
                            ax.plot(df_cat[x_col], p(numeric_x), "--", color=line_color)

                        if should_show_labels(period_type, line_count):
                            for i in range(1, len(df_cat)):
                                if pd.notnull(df_cat['Pct_Change'].iloc[i]):
                                    ax.text(df_cat[x_col].iloc[i], df_cat['Adjusted Net Value'].iloc[i],
                                            f"{df_cat['Pct_Change'].iloc[i]:.1f}%", ha="center", va="bottom", fontsize=9, color="red")

                    ax.set_title(f"{period.capitalize()}ly Sales by {category} (Cats: {chosen_categories_str})", fontsize=16)
                    ax.set_ylabel("Adjusted Net Value", fontsize=14)
                    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
                    ax.tick_params(axis="both", which="major", labelsize=12)
                    
                    if should_show_legend(line_count):
                        if ax.get_legend() is not None:
                            ax.legend(title=category, loc="upper left", bbox_to_anchor=(1, 1), fontsize=12)
                    else:
                        if ax.get_legend() is not None:
                            ax.get_legend().remove()

                    plt.tight_layout()
                    if save_figures:
                        fname = sanitize_filename(f"{period.capitalize()}ly_Sales_by_{category}.png")
                        plt.savefig(figure_dir / fname, bbox_inches="tight")
                    if enable_plotting:
                        plt.show()
                    else:
                        plt.close()
                    if download_data:
                        sort_col = {'year': 'Year', 'quarter': 'Quarter_Num_Total', 'month': 'Month.Year.Date'}[period_type]
                        out_df = period_df.copy().sort_values(by=sort_col, ascending=False)
                        csv_name = sanitize_filename(f"{period.capitalize()}ly_Sales_by_{category}_Data.csv")
                        figure_dir.mkdir(parents=True, exist_ok=True)
                        out_df.to_csv(figure_dir / csv_name, index=False)
                        print(f"Raw data saved to {csv_name}")
                available_graphs.append((f"{period.capitalize()}ly Sales Trend by {category}", plot_category_sales))
    
    # --- Quantity-based Graphs for Core Code and Core Code 4 ---
    for category in ['Core Code', 'Core Code 4']:
        quantity_key = f"{category}_quantity"
        if quantity_key in grouped_dfs:
            data_dict = grouped_dfs[quantity_key]
            
            # Pie chart for overall quantity distribution by attribute category
            if "overall" in data_dict and not data_dict["overall"].empty:
                def plot_quantity_pie_chart(category=category, overall_df=data_dict["overall"].copy()):
                    if merge_small_sections:
                        total_quantity = overall_df["Quantity"].sum()
                        if total_quantity > 0:
                            overall_df["Percentage"] = (overall_df["Quantity"] / total_quantity) * 100
                        else:
                            overall_df["Percentage"] = 0
                        
                        if overall_df[overall_df["Percentage"] < other_threshold].shape[0] > 0:
                            overall_df[category] = overall_df.apply(
                                lambda row: "Other" if row["Percentage"] < other_threshold else row[category], axis=1
                            )
                            agg_spec = {'Quantity': 'sum'}
                            if 'Adjusted Net Value' in overall_df.columns:
                                agg_spec['Adjusted Net Value'] = 'sum'
                            if 'Core Code Item Description' in overall_df.columns:
                                agg_spec['Core Code Item Description'] = 'first'
                            if 'Core Code 4 Family' in overall_df.columns:
                                agg_spec['Core Code 4 Family'] = 'first'
                            overall_df = overall_df.groupby(category, as_index=False).agg(agg_spec)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sum_val = overall_df["Quantity"].sum()
                    if sum_val == 0:
                        plt.title(f"No quantity sold for {category}.")
                        if enable_plotting:
                            plt.show()
                        else:
                            plt.close()
                        return
                    
                    overall_df["Percentage"] = overall_df["Quantity"] / sum_val * 100
                    overall_df.sort_values("Quantity", ascending=False, inplace=True)
                    
                    # Add family names and descriptions to labels if available
                    labels = []
                    family_col = f'{category} Family'
                    desc_col  = f'{category} Item Description'
                    for _, row in overall_df.iterrows():
                        label = str(row[category])
                        # Prefer the item description; fall back to family
                        if category == 'Core Code' and desc_col in row and pd.notna(row[desc_col]):
                            label = f"{row[category]} - {row[desc_col]}"
                        elif family_col in row and pd.notna(row[family_col]):
                            label = f"{row[category]} ({row[family_col]})"
                        labels.append(label)
                    
                    ax.pie(overall_df["Quantity"],
                           labels=labels,
                           autopct="%1.1f%%", startangle=140)
                    ax.set_title(f"Quantity Distribution by {category}\n(Cats: {chosen_categories_str})", fontsize=16)
                    plt.tight_layout()
                    
                    if save_figures:
                        fname = f"Quantity_Distribution_by_{category}.png"
                        plt.savefig(figure_dir / fname, bbox_inches="tight")
                    if enable_plotting:
                        plt.show()
                    else:
                        plt.close()
                    
                    if download_data:
                        out_data = overall_df.copy().sort_values("Quantity", ascending=False)

                        # Ensure the item description column is merged in for Core Code
                        if category == 'Core Code' and valid_items is not None:
                            desc_col = "Core Code Item Description"
                            if desc_col not in out_data.columns and desc_col in valid_items.columns:
                                desc_map = valid_items[['Core Code', desc_col]].drop_duplicates()
                                out_data = out_data.merge(desc_map, on='Core Code', how="left")

                        fname = sanitize_filename(f"Quantity_Distribution_by_{category}_Data.csv")
                        figure_dir.mkdir(parents=True, exist_ok=True)
                        out_data.to_csv(figure_dir / fname, index=False)
                        print(f"Raw data for {category} quantity distribution saved to {fname}")
                
                available_graphs.append((f"Quantity Distribution by {category} (Overall)", plot_quantity_pie_chart))
            
            # Line charts by period for quantity
            for period in ["year", "quarter", "month"]:
                if period in data_dict and not data_dict[period].empty:
                    # Create a closure that captures the current values
                    def make_quantity_plot(current_period, current_data_dict, current_category, color_maps):
                        def plot_quantity_sales():
                            period_df = current_data_dict[current_period].copy()
                            
                            if current_period == "quarter":
                                period_df["Quarter"] = period_df["Quarter"].astype(str)
                                period_df["Year"] = period_df["Quarter"].str[:4].astype(int)
                                period_df["Quarter_Num"] = period_df["Quarter"].str[-1].astype(int)
                                period_df["Quarter_Num_Total"] = period_df["Year"] * 4 + period_df["Quarter_Num"] - 1
                                x_col = "Quarter_Num_Total"
                                period_type = "quarter"
                                period_df.sort_values("Quarter_Num_Total", inplace=True)
                            elif current_period == "month":
                                period_df["Month.Year.Date"] = pd.to_datetime(period_df["Month.Year.Date"], errors='coerce')
                                x_col = "Month.Year.Date"
                                period_type = "month"
                                period_df.sort_values("Month.Year.Date", inplace=True)
                            else:
                                x_col = "Year"
                                period_type = "year"
                                period_df.sort_values("Year", inplace=True)

                            unique_cats = period_df[current_category].unique()
                            line_count = len(unique_cats)
                            fig, ax = plt.subplots(figsize=(14, 7) if current_period == "month" else (12, 7))
                            
                            if current_period == "quarter":
                                ax.set_xticks(period_df["Quarter_Num_Total"].unique())
                                ax.set_xticklabels(period_df.drop_duplicates("Quarter_Num_Total")["Quarter"], rotation=45)
                                for yr in sorted(period_df["Year"].unique())[:-1]:
                                    last_q = period_df.loc[period_df["Year"] == yr, "Quarter_Num_Total"].max()
                                    ax.axvline(x=last_q + 0.5, color="gray", linestyle="--", alpha=0.7)
                            
                            palette_map = None
                            if color_maps and current_category in color_maps:
                                palette_map = color_maps[current_category]

                            for cat_val in unique_cats:
                                df_cat = period_df[period_df[current_category].isin([cat_val])].copy()
                                if df_cat.empty:
                                    continue
                                df_cat['Pct_Change'] = df_cat['Quantity'].pct_change() * 100
                                df_cat['Pct_Change'].replace([float("inf"), float("-inf")], np.nan, inplace=True)
                                
                                display_name = str(cat_val)
                                family_col = f'{current_category} Family'
                                desc_col   = f'{current_category} Item Description'

                                if current_category == 'Core Code' and desc_col in df_cat.columns and pd.notna(df_cat[desc_col].iloc[0]):
                                    display_name = f"{cat_val} - {df_cat[desc_col].iloc[0]}"
                                elif family_col in df_cat.columns and pd.notna(df_cat[family_col].iloc[0]):
                                    display_name = f"{cat_val} ({df_cat[family_col].iloc[0]})"
                                
                                line_color = palette_map.get(cat_val) if palette_map else None

                                sns.lineplot(data=df_cat, x=x_col, y="Quantity",
                                           marker="o", linewidth=2.0, label=display_name, ax=ax, color=line_color)
                                
                                if line_count <= max_trendlines and len(df_cat) > 1:
                                    numeric_x = mdates.date2num(df_cat[x_col]) if period_type == 'month' else df_cat[x_col]
                                    z = np.polyfit(numeric_x, df_cat["Quantity"], 1)
                                    p = np.poly1d(z)
                                    ax.plot(df_cat[x_col], p(numeric_x), "--", color=line_color)
                                
                                if should_show_labels(period_type, line_count):
                                    for i in range(1, len(df_cat)):
                                        if pd.notnull(df_cat['Pct_Change'].iloc[i]):
                                            ax.text(df_cat[x_col].iloc[i], df_cat['Quantity'].iloc[i],
                                                   f"{df_cat['Pct_Change'].iloc[i]:.1f}%", ha="center", va="bottom", fontsize=9, color="red")
                            
                            ax.set_title(f"{current_period.capitalize()}ly Quantity Sold by {current_category} (Cats: {chosen_categories_str})", fontsize=16)
                            ax.set_ylabel("Quantity Sold", fontsize=14)
                            ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
                            ax.tick_params(axis="both", which="major", labelsize=12)
                            
                            if should_show_legend(line_count):
                                if ax.get_legend() is not None:
                                    ax.legend(title=current_category, loc="upper left", bbox_to_anchor=(1, 1), fontsize=12)
                            else:
                                if ax.get_legend() is not None:
                                    ax.get_legend().remove()
                            
                            plt.tight_layout()
                            
                            if save_figures:
                                fname = sanitize_filename(f"{current_period.capitalize()}ly_Quantity_by_{current_category}.png")
                                plt.savefig(figure_dir / fname, bbox_inches="tight")
                            
                            if enable_plotting:
                                plt.show()
                            else:
                                plt.close()
                            
                            if download_data:
                                sort_col = {'year': 'Year', 'quarter': 'Quarter_Num_Total', 'month': 'Month.Year.Date'}[period_type]
                                out_df = period_df.copy().sort_values(by=sort_col, ascending=False)
                                
                                if current_category == 'Core Code' and valid_items is not None:
                                    desc_col = "Core Code Item Description"
                                    if desc_col not in out_df.columns and desc_col in valid_items.columns:
                                        desc_map = valid_items[['Core Code', desc_col]].drop_duplicates()
                                        out_df = out_df.merge(desc_map, on='Core Code', how="left")

                                csv_name = sanitize_filename(f"{current_period.capitalize()}ly_Quantity_by_{current_category}_Data.csv")
                                figure_dir.mkdir(parents=True, exist_ok=True)
                                out_df.to_csv(figure_dir / csv_name, index=False)
                                print(f"Raw data saved to {csv_name}")
                        
                        return plot_quantity_sales
                    
                    plot_func = make_quantity_plot(period, data_dict, category, color_maps)
                    available_graphs.append((f"{period.capitalize()}ly Quantity Trend by {category}", plot_func))
    
    return available_graphs

# --- Interactive Menu & Helper Functions ---

def interactive_menu(earliest_date, latest_date, default_start, initial_params):
    params = initial_params.copy()

    # Ensure data-dependent date ranges are set correctly at the start
    params["earliest_date"] = earliest_date
    params["latest_date"] = latest_date
    params["default_start"] = default_start
    if params.get("start_date") is None:
        params["start_date"] = default_start

    menu_text = """
--- Interactive Menu ---
1.  Enter Start Date (Current: {start_date_str})
2.  Enter End Date   (Current: {end_date_str})
3.  Enter Product Codes or Ranges (comma-separated) (Current: {product_codes})
4.  Edit Contamination Parameter (Current: {contamination})
5.  Edit Max Trendlines (Current: {max_trendlines})
6.  Toggle Data Labels (Yearly: {show_labels_yearly}, Quarterly: {show_labels_quarterly}, Monthly: {show_labels_monthly})
7.  Edit 'Max Lines for Labels' (Current: {max_lines_for_labels})
8.  Edit 'Max Lines for Legend' (Current: {max_lines_for_legend})
9.  Edit 'Other' Threshold for Pie Charts (Current: {other_threshold})
10. Toggle Info Box Options (Yearly={show_info_box_yearly}, Quarterly={show_info_box_quarterly}, Monthly={show_info_box_monthly}, max_lines={max_lines_for_info_box})
11. Toggle Advanced Attribute Filtering (Current: {enable_advanced_filtering})
12. Toggle Save Figures (Current: {save_figures})
13. Toggle Skip run_transformation (Current: {skip_transformation})
14. Toggle Download Data (Current: {download_data})
15. Toggle Enable Plotting (Current: {enable_plotting})
--- Profile Management ---
16. Save Current Parameters to Profile
17. Load Profile
18. Restore Default Parameters
19. Proceed to Filtering & Charting
20. Exit Program
"""
    while True:
        # Prepare display strings for dates
        display_params = params.copy()
        display_params['start_date_str'] = params['start_date'].strftime('%Y-%m-%d') if params.get('start_date') else 'Not Set'
        display_params['end_date_str'] = params['end_date'].strftime('%Y-%m-%d') if params.get('end_date') else 'Not Set'

        print(menu_text.format(**display_params))
        choice = input("Select an option (1-20): ").strip()
        if choice == '1':
            user_date = input(f"Enter Start Date (YYYY-MM-DD), earliest is {earliest_date.strftime('%Y-%m-%d')}: ").strip()
            try:
                if user_date:
                    params["start_date"] = pd.to_datetime(user_date)
            except Exception:
                print("Invalid date format.")
        elif choice == '2':
            user_date = input(f"Enter End Date (YYYY-MM-DD), latest is {latest_date.strftime('%Y-%m-%d')}: ").strip()
            try:
                if user_date:
                    params["end_date"] = pd.to_datetime(user_date)
            except Exception:
                print("Invalid date format.")
        elif choice == '3':
            codes_str = input("Enter product codes or ranges, comma-separated (blank=none): ").strip()
            params["product_codes"] = [s.strip() for s in codes_str.split(',') if s.strip()] if codes_str else None
        elif choice == '4':
            cont_str = input("Enter contamination (e.g. 0.01): ").strip()
            try:
                params["contamination"] = float(cont_str)
            except ValueError:
                print("Invalid numeric format.")
        elif choice == '5':
            mt_str = input("Enter max trendlines (integer): ").strip()
            try:
                params["max_trendlines"] = int(mt_str)
            except ValueError:
                print("Invalid integer.")
        elif choice == '6':
            sub = input("Toggle labels for which? (y for Year, q for Quarter, m for Month, all): ").strip().lower()
            if sub in ['y', 'year', 'yearly']:
                params['show_labels_yearly'] = not params['show_labels_yearly']
            elif sub in ['q', 'quarter', 'quarterly']:
                params['show_labels_quarterly'] = not params['show_labels_quarterly']
            elif sub in ['m', 'month', 'monthly']:
                params['show_labels_monthly'] = not params['show_labels_monthly']
            elif sub == 'all':
                params['show_labels_yearly'] = not params['show_labels_yearly']
                params['show_labels_quarterly'] = not params['show_labels_quarterly']
                params['show_labels_monthly'] = not params['show_labels_monthly']
        elif choice == '7':
            label_str = input("Enter max lines for labels (integer): ").strip()
            try:
                params["max_lines_for_labels"] = int(label_str)
            except ValueError:
                print("Invalid integer.")
        elif choice == '8':
            legend_str = input("Enter max lines for legend (integer): ").strip()
            try:
                params["max_lines_for_legend"] = int(legend_str)
            except ValueError:
                print("Invalid integer.")
        elif choice == '9':
            oth_str = input("Enter 'Other' threshold for pie charts (float): ").strip()
            try:
                params["other_threshold"] = float(oth_str)
            except ValueError:
                print("Invalid float.")
        elif choice == '10':
            sub_option = input("Toggle Info Box for which? (y/q/m/max/all): ").strip().lower()
            if sub_option in ['y', 'year', 'yearly']:
                params['show_info_box_yearly'] = not params['show_info_box_yearly']
            elif sub_option in ['q', 'quarter', 'quarterly']:
                params['show_info_box_quarterly'] = not params['show_info_box_quarterly']
            elif sub_option in ['m', 'month', 'monthly']:
                params['show_info_box_monthly'] = not params['show_info_box_monthly']
            elif sub_option == 'max':
                lines_str = input("Enter new max lines for info box: ").strip()
                if lines_str.isdigit():
                    params['max_lines_for_info_box'] = int(lines_str)
            elif sub_option == 'all':
                params['show_info_box_yearly'] = not params['show_info_box_yearly']
                params['show_info_box_quarterly'] = not params['show_info_box_quarterly']
                params['show_info_box_monthly'] = not params['show_info_box_monthly']
        elif choice == '11':
            params["enable_advanced_filtering"] = not params["enable_advanced_filtering"]
            print(f"Advanced filtering set to {params['enable_advanced_filtering']}")
        elif choice == '12':
            params["save_figures"] = not params["save_figures"]
            print(f"Save Figures set to {params['save_figures']}")
        elif choice == '13':
            params["skip_transformation"] = not params["skip_transformation"]
            print(f"Skip run_transformation set to {params['skip_transformation']}")
        elif choice == '14':
            params["download_data"] = not params["download_data"]
            print(f"Download Data set to {params['download_data']}")
        elif choice == '15':
            params["enable_plotting"] = not params["enable_plotting"]
            print(f"Enable plotting set to {params['enable_plotting']}")
        elif choice == '16':
            profile_name = input("Enter a name for this profile (e.g., 'monthly_report'): ").strip()
            if profile_name:
                save_params(params, profile_name)
            else:
                print("Profile name cannot be empty.")
        elif choice == '17':
            if not CONFIG_FILE.exists():
                print("No config file found. Save a profile first.")
                continue
            try:
                with open(CONFIG_FILE, 'r') as f:
                    config_data = json.load(f)
                available_profiles = list(config_data.get("profiles", {}).keys())
                if not available_profiles:
                    print("No saved profiles found.")
                    continue
                
                print("Available profiles:", ", ".join(available_profiles))
                profile_to_load = input("Enter profile name to load: ").strip()

                if profile_to_load in available_profiles:
                    # Load the selected profile's settings
                    loaded_p = load_params(get_default_params(), profile_to_load)
                    # Preserve the data-dependent date ranges from the current session
                    loaded_p['earliest_date'] = params['earliest_date']
                    loaded_p['latest_date'] = params['latest_date']
                    loaded_p['default_start'] = params['default_start']
                    params.clear()
                    params.update(loaded_p)
                    print(f"Profile '{profile_to_load}' loaded.")
                else:
                    print("Invalid profile name.")
            except Exception as e:
                print(f"Error loading profiles: {e}")

        elif choice == '18':
            print("Restoring default parameters...")
            defaults = get_default_params()
            # Preserve the data-dependent date ranges
            defaults['earliest_date'] = params['earliest_date']
            defaults['latest_date'] = params['latest_date']
            defaults['default_start'] = params['default_start']
            # Reset start/end dates to their initial state for this data
            defaults['start_date'] = params['default_start']
            defaults['end_date'] = None
            params.clear()
            params.update(defaults)
            print("Defaults restored. Start and end dates have been reset.")
        elif choice == '19':
            if params.get("start_date") is None or params.get("end_date") is None:
                print("Please set both start and end dates before proceeding.")
            else:
                return params
        elif choice == '20':
            return None # Signal to exit the program
        else:
            print("Invalid choice. Try again.")

def choose_categories_from_menu(valid_items: pd.DataFrame):
    categories = sorted(valid_items['Product Category'].dropna().unique().tolist())
    if not categories:
        print("No categories found. Skipping category filtering.")
        return None
    while True:
        print("\n--- Choose Categories to Whitelist ---")
        for idx, cat in enumerate(categories, start=1):
            print(f"{idx}. {cat}")
        user_input = input("Enter category numbers (comma-separated) or 'all'/'none'/'back'): ").strip().lower()
        if user_input in ['none', '']:
            print("Skipping category filtering.")
            return None
        elif user_input == 'all':
            print("All categories selected.")
            return categories
        elif user_input == 'back':
            return 'BACK'
        else:
            try:
                selected_nums = sorted({int(part) for part in user_input.split(',') if part.strip().isdigit()})
            except ValueError:
                selected_nums = []
            if not selected_nums:
                print("No valid selections. Try again.")
                continue
            chosen = [categories[i - 1] for i in selected_nums if 1 <= i <= len(categories)]
            print(f"Chosen categories: {chosen}")
            return chosen

def choose_attribute_values_from_menu(valid_items: pd.DataFrame) -> dict:
    attributes = ["Size", "Finish", "Colour", "Region", "Glass", "Type", "IP44"]
    whitelists = {}
    print("\nWhich attributes do you want to filter? (e.g. '1,3,5') or type 'skip' or 'back'")
    for idx, attr in enumerate(attributes, start=1):
        print(f"{idx}. {attr}")
    choice = input("Choice: ").strip().lower()
    if choice in ['skip', '', 'none']:
        return {}
    if choice == 'back':
        return 'BACK'
    try:
        selected_idx = sorted({int(part) for part in choice.split(',') if part.strip().isdigit()})
    except ValueError:
        selected_idx = []
    if not selected_idx:
        print("No valid attribute selections. Skipping advanced filtering.")
        return {}
    for idx in selected_idx:
        if not (1 <= idx <= len(attributes)):
            continue
        attribute_name = attributes[idx - 1]
        
        # Get unique non-NA values and check for the presence of NA values
        unique_vals = sorted(valid_items[attribute_name].dropna().unique().tolist())
        has_na = valid_items[attribute_name].isna().any()

        if not unique_vals and not has_na:
            print(f"No values for attribute '{attribute_name}'. Skipping it.")
            continue
            
        while True:
            print(f"\n--- Whitelist for {attribute_name} ---")
            for i, val in enumerate(unique_vals, start=1):
                print(f"{i}. {val}")
            
            na_option_num = 0
            if has_na:
                na_option_num = len(unique_vals) + 1
                print(f"{na_option_num}. Not Specified (N/A)")

            user_in = input("Enter selection (or 'all'/'none'/'back'): ").strip().lower()
            
            if user_in in ['none', '']:
                print(f"Skipping {attribute_name}.")
                break
            elif user_in == 'all':
                print(f"Using all values for {attribute_name}.")
                all_values = unique_vals
                if has_na:
                    all_values.append(np.nan)
                whitelists[attribute_name] = all_values
                break
            elif user_in == 'back':
                return 'BACK' 
            else:
                try:
                    chosen_nums = sorted({int(part) for part in user_in.split(',') if part.strip().isdigit()})
                except ValueError:
                    chosen_nums = []
                
                if not chosen_nums:
                    print("No valid selection. Try again.")
                    continue

                chosen_vals = []
                for num in chosen_nums:
                    if 1 <= num <= len(unique_vals):
                        chosen_vals.append(unique_vals[num - 1])
                    elif has_na and num == na_option_num:
                        chosen_vals.append(np.nan)
                
                if chosen_vals:
                    whitelists[attribute_name] = chosen_vals
                
                break
    return whitelists

def choose_graphs_from_menu(available_graphs: list) -> list:
    """Let user choose which graphs to plot from the available options."""
    if not available_graphs:
        return []
    
    print(f"\nAvailable Charts ({len(available_graphs)} total):")
    print("="*50)
    for i, (graph_title, _) in enumerate(available_graphs, 1):
        print(f"{i:2d}. {graph_title}")
    
    print("\nSelection Options:")
    print("  - Enter numbers (e.g., '1,3,5' or '1-5')")
    print("  - Enter 'all' to select all charts")
    print("  - Enter 'back' to return to the filtering options")
    
    user_input = input("\nSelect charts to plot: ").strip().lower()
    
    if user_input in ['back', 'b']:
        return None # Signal to go back
    
    if not user_input or user_input == 'all':
        return available_graphs
    
    selected_indices = set()
    
    for part in user_input.split(','):
        part = part.strip()
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                selected_indices.update(range(start, end + 1))
            except ValueError:
                print(f"Invalid range format: {part}")
        else:
            try:
                selected_indices.add(int(part))
            except ValueError:
                print(f"Invalid number: {part}")
    
    valid_indices = [i for i in selected_indices if 1 <= i <= len(available_graphs)]
    if not valid_indices:
        print("No valid selections. Plotting all charts by default.")
        return available_graphs
    
    selected_graphs = [available_graphs[i-1] for i in sorted(valid_indices)]
    print(f"Selected {len(selected_graphs)} charts.")
    return selected_graphs



@contextmanager
def timeout(seconds: int):
    def _handler(signum, frame):
        raise TimeoutError
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

if __name__ == "__main__":
    # --- Initial Setup: Load data once to determine date ranges ---
    file_names = ["AllData 15.07.25"]
    combined_for_range = pd.DataFrame()
    for file_name in file_names:
        file_path = data_dir / f"{file_name}.csv"
        df_temp = load_and_prepare_data(file_path)
        combined_for_range = pd.concat([combined_for_range, df_temp], ignore_index=True)
    overall_earliest = combined_for_range['SalesOrder.Date'].min()
    overall_latest = combined_for_range['SalesOrder.Date'].max()
    default_start = overall_latest - pd.DateOffset(years=4) if overall_latest else None

    # Load parameters from the last session, or use defaults
    current_params = load_params(get_default_params())
    
    # --- Data Caching ---
    valid_items_cache = None
    file_name_for_processing = file_names[0]

    # --- Main Application Loop ---
    while True:
        # --- STATE 1: MAIN MENU & PARAMETER SETUP ---
        user_params = interactive_menu(overall_earliest, overall_latest, default_start, current_params)
        
        if user_params is None: # User chose to exit the program
            print("Exiting program.")
            break
        
        current_params = user_params

        # --- STATE 2: DATA PROCESSING (based on main menu params) ---
        # Get the latest dates from the parameters after the menu selection
        start_date = current_params["start_date"]
        end_date = current_params["end_date"]
        
        if valid_items_cache is None:
            print("\nProcessing data based on selected parameters...")
            
            if current_params["skip_transformation"]:
                useable_file = data_dir / "Useable Data.csv"
                valid_items_base = pd.read_csv(useable_file)
            else:
                file_path = data_dir / f"{file_name_for_processing}.csv"
                mappings_path = data_dir / 'Unique Product Code Values.csv'
                mappings_df = load_mappings_file(mappings_path)
                valid_product_codes_path = data_dir / 'allproductcodes2.csv'
                valid_product_codes_df = pd.read_csv(valid_product_codes_path)
                valid_product_codes_df['Product Codes'] = valid_product_codes_df['Product Codes'].str.split('/').str[0]
                valid_product_codes_list = valid_product_codes_df['Product Codes'].tolist()
                
                combined_raw = load_and_prepare_data(file_path)
                valid_items_processed, _ = process_data(combined_raw, file_name_for_processing, valid_product_codes_list, mappings_df)
                
                temp_path = data_dir / "valid_items_file.csv"
                valid_items_processed.to_csv(temp_path, index=False)

                old_to_new_path = data_dir / "Old to New Product Codes.csv"
                old_new_raw = pd.read_csv(old_to_new_path)
                new_code_unique_path = data_dir / "New Code Unique Product Code Values.csv"
                new_code_unique_df = load_mappings_file(new_code_unique_path)
                
                valid_items_base = run_transformation(str(temp_path), old_new_raw, mappings_df, new_code_unique_df)
                
                out_path = data_dir / "Useable Data.csv"
                valid_items_base.to_csv(out_path, index=False)

            # Ensure the 'Date' column is in datetime format before caching
            valid_items_base['Date'] = pd.to_datetime(valid_items_base['Date'], errors='coerce')
            valid_items_cache = valid_items_base.copy()
        else:
            print("\nUsing cached data.")
            valid_items_base = valid_items_cache.copy()

        # Filter by date range selected in the main menu
        valid_items_dated = valid_items_base[(valid_items_base['Date'] >= start_date) & (valid_items_base['Date'] <= end_date)].copy()
        print(f"Data processed. {len(valid_items_dated)} records in the selected date range.")

        # Create the figure directory for this session ONCE.
        figure_directory = create_new_figure_directory(
            figure_dir, file_name_for_processing,
            current_params["save_figures"], current_params["download_data"]
        )

        # --- STATE 3: REFINEMENT & CHARTING LOOP ---
        # This loop allows users to refine filters and re-plot without full data reprocessing
        while True:
            # --- Category Filtering ---
            chosen_categories = choose_categories_from_menu(valid_items_dated)
            if chosen_categories == 'BACK':
                break # Exit this loop to go back to the main menu

            filtered_items = valid_items_dated.copy()
            if chosen_categories:
                filtered_items = filtered_items[filtered_items['Product Category'].isin(chosen_categories)]

            # --- Attribute Filtering ---
            if current_params["enable_advanced_filtering"]:
                attribute_whitelists = choose_attribute_values_from_menu(filtered_items)
                if attribute_whitelists == 'BACK':
                    continue # Go back to the start of this loop (category selection)
                filtered_items = apply_attribute_whitelists(filtered_items, attribute_whitelists)

            # --- Final Filtering & Grouping ---
            if current_params["product_codes"]:
                filtered_items = filter_by_codes_and_ranges(filtered_items, current_params["product_codes"])
            
            non_outlier, _ = remove_upper_outliers_quantile_isolation_forest(
                filtered_items, remove_outliers=True, 
                column_name='Adjusted Net Value', contamination=current_params["contamination"]
            )

            # --- MODIFICATION: Create consistent color maps for specific categories ---
            color_map_categories = ['Size', 'Finish', 'Region', 'Type']
            color_maps = {}
            for cat in color_map_categories:
                if cat in non_outlier.columns:
                    unique_values = non_outlier[cat].unique()
                    has_nan = any(pd.isna(v) for v in unique_values)
                    other_values = sorted([v for v in unique_values if pd.notna(v)])
                    
                    sorted_unique_values = other_values
                    if has_nan:
                        sorted_unique_values.append(np.nan)

                    palette = sns.color_palette('husl', len(sorted_unique_values))
                    
                    color_map = {}
                    for i, value in enumerate(sorted_unique_values):
                        # Use a consistent key for NaN to handle lookups
                        key = 'Not Specified' if pd.isna(value) else value
                        color_map[key] = palette[i]
                    color_maps[cat] = color_map
            
            # When grouping, replace NaN with 'Not Specified' to match color map keys
            for cat in color_map_categories:
                if cat in non_outlier.columns:
                    non_outlier[cat] = non_outlier[cat].fillna('Not Specified')


            grouped_dfs = create_grouped_dataframes(non_outlier)
            stats_results = compute_statistics(grouped_dfs)

            chosen_categories_str = "All" if not chosen_categories else ", ".join(chosen_categories)
            # ... (add other filter info to chosen_categories_str if needed) ...

            # Define the list of keys that are valid for create_charts
            chart_param_keys = [
                'save_figures', 'max_trendlines', 
                'other_threshold', 'show_labels_yearly', 'show_labels_quarterly', 
                'show_labels_monthly', 'max_lines_for_labels', 'max_lines_for_legend', 
                'show_info_box_yearly', 'show_info_box_quarterly', 'show_info_box_monthly', 
                'max_lines_for_info_box', 'download_data', 'enable_plotting'
            ]
            # Create a new dictionary with only the valid keys
            chart_params = {key: current_params[key] for key in chart_param_keys if key in current_params}

            available_graphs = create_charts(
                grouped_dfs, stats_results, figure_directory,
                **chart_params, # Pass only the relevant chart parameters
                chosen_categories_str=chosen_categories_str,
                valid_items=filtered_items,
                color_maps=color_maps, # Pass the generated color maps
            )

            # --- Chart Selection Loop ---
            while True:
                if not available_graphs:
                    print("No graphs can be generated with the current filters.")
                    break

                selected_graphs = choose_graphs_from_menu(available_graphs)
                if selected_graphs is None: # User chose to go back
                    break # Exit chart loop, will re-enter the filter loop

                if not selected_graphs:
                    print("No charts selected.")
                else:
                    mode_msg = "display and save files" if current_params["enable_plotting"] else "save files only"
                    print(f"\nProcessing {len(selected_graphs)} chart(s) → {mode_msg}…")
                    for graph_title, plot_func in selected_graphs:
                        print(f"  • {graph_title}")
                        plot_func()
                    if current_params["download_data"]:
                        print("CSV exports completed.")
            
            # After chart loop breaks, ask user what to do next
            nav_choice = input("\nGo back to [F]iltering, or return to the [M]ain Menu? (F/M): ").strip().lower()
            if nav_choice == 'm':
                break # Breaks the filtering loop, will go back to the main menu
            else:
                continue # Continues the filtering loop (re-starts at category selection)
