from pathlib import Path
import hashlib
import os
import tempfile

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
DEFAULT_CSV = DATA_DIR / "transportation_fuel_analysis_messy.csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="FuelScope | Vehicle Intelligence", page_icon="🚘", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #f4f7fb; }
[data-testid="stSidebar"] { background: #111c32; }
[data-testid="stSidebar"] * { color: #e8eef8 !important; }
[data-testid="stSidebar"] .stRadio label { padding: 7px 4px; }
.block-container { padding-top: 1.8rem; padding-bottom: 2rem; max-width: 1500px; }
.hero { background: linear-gradient(120deg,#14213d 0%,#213b67 65%,#167d87 100%); padding: 28px 32px; border-radius: 18px; color: white; margin-bottom: 22px; box-shadow: 0 10px 28px rgba(20,33,61,.12); }
.hero h1 { font-family: Manrope,sans-serif; font-size: 30px; margin: 0 0 8px 0; font-weight: 800; color: white; }
.hero p { color: #d8e5f6; margin: 0; font-size: 14px; }
.eyebrow { text-transform: uppercase; letter-spacing: 2px; font-size: 10px; font-weight: 700; color: #8ee3db; margin-bottom: 10px; }
.kpi { background: white; padding: 18px 20px; border: 1px solid #e5ebf3; border-radius: 14px; min-height: 116px; box-shadow: 0 4px 14px rgba(25,48,80,.035); }
.kpi-label { font-size: 12px; color: #718096; font-weight: 600; margin-bottom: 9px; }
.kpi-value { font-family: Manrope,sans-serif; color: #172641; font-size: 25px; font-weight: 800; line-height: 1.2; }
.kpi-note { color: #8794a8; font-size: 11px; margin-top: 8px; }
.section-title { font-family: Manrope,sans-serif; color: #172641; font-size: 19px; font-weight: 800; margin: 12px 0 3px; }
.section-sub { color: #7a879a; font-size: 12px; margin-bottom: 14px; }
.panel { background: white; border: 1px solid #e5ebf3; border-radius: 14px; padding: 16px 18px; }
.small-note { color: #75839a; font-size: 12px; }
div[data-testid="stMetric"] { background: white; border: 1px solid #e5ebf3; padding: 15px 18px; border-radius: 12px; }
.stButton>button { border-radius: 9px; font-weight: 600; }
hr { border-color: #e5ebf3; }
</style>
""", unsafe_allow_html=True)


def make_spark():
    return (SparkSession.builder.appName("FuelScopeTransportationFuelETL")
            .master("local[*]")
            .config("spark.ui.enabled", "false")
            .config("spark.driver.host", "127.0.0.1")
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.sql.execution.arrow.pyspark.enabled", "false")
            .getOrCreate())


@st.cache_data(show_spinner=False, max_entries=4)
def process_dataset(csv_path: str, file_signature: str):
    spark = make_spark()
    spark.sparkContext.setLogLevel("ERROR")
    try:
        raw = spark.read.option("header", True).option("inferSchema", True).option("mode", "PERMISSIVE").csv(csv_path)
        expected = ["vehicle_year", "make", "model", "vehicle_class", "drive", "transmission", "engine_cylinders", "engine_displacement_l", "fuel_type", "city_mpg", "highway_mpg", "combined_mpg", "annual_fuel_cost_usd", "co2_grams_per_mile"]
        missing_columns = [c for c in expected if c not in raw.columns]
        if missing_columns:
            raise ValueError("The CSV is missing required columns: " + ", ".join(missing_columns))
        raw = raw.select(*expected)
        original_rows = raw.count()
        original_columns = len(raw.columns)
        duplicate_count = original_rows - raw.dropDuplicates().count()
        missing_before = sum(raw.filter(F.col(c).isNull() | (F.trim(F.col(c).cast("string")) == "")).count() for c in raw.columns)
        invalid_year = raw.filter((F.col("vehicle_year").cast("int") < 1900) | (F.col("vehicle_year").cast("int") > 2026)).count()
        invalid_mpg = raw.filter((F.col("city_mpg").cast("double") <= 0) | (F.col("highway_mpg").cast("double") <= 0) | (F.col("combined_mpg").cast("double") <= 0)).count()
        invalid_cost = raw.filter(F.col("annual_fuel_cost_usd").cast("double") < 0).count()
        invalid_co2 = raw.filter(F.col("co2_grams_per_mile").cast("double") < 0).count()

        df = raw.dropDuplicates()
        text_columns = ["make", "model", "vehicle_class", "drive", "transmission", "fuel_type"]
        numeric_columns = ["vehicle_year", "engine_cylinders", "engine_displacement_l", "city_mpg", "highway_mpg", "combined_mpg", "annual_fuel_cost_usd", "co2_grams_per_mile"]
        for c in text_columns:
            df = df.withColumn(c, F.trim(F.col(c).cast("string")))
            df = df.withColumn(c, F.when(F.col(c) == "", None).otherwise(F.col(c)))
        for c in numeric_columns:
            df = df.withColumn(c, F.col(c).cast("double"))
        df = df.withColumn("vehicle_year", F.when((F.col("vehicle_year") < 1900) | (F.col("vehicle_year") > 2026), None).otherwise(F.col("vehicle_year")))
        for c in ["engine_cylinders", "engine_displacement_l"]:
            df = df.withColumn(c, F.when(F.col(c) < 0, None).otherwise(F.col(c)))
        for c in ["city_mpg", "highway_mpg", "combined_mpg"]:
            df = df.withColumn(c, F.when(F.col(c) <= 0, None).otherwise(F.col(c)))
        for c in ["annual_fuel_cost_usd", "co2_grams_per_mile"]:
            df = df.withColumn(c, F.when(F.col(c) < 0, None).otherwise(F.col(c)))
        for c in ["make", "model", "vehicle_class", "drive", "transmission"]:
            df = df.withColumn(c, F.initcap(F.lower(F.col(c))))
        fuel = F.lower(F.trim(F.col("fuel_type")))
        df = df.withColumn("fuel_type", F.when(fuel.contains("electric") | fuel.contains("hybrid"), "Electricity / Hybrid")
                           .when(fuel.contains("e85") | fuel.contains("flex"), "E85 / Flex Fuel")
                           .when(fuel.contains("diesel"), "Diesel")
                           .when(fuel.contains("cng"), "CNG")
                           .when(fuel.contains("premium"), "Premium")
                           .when(fuel.contains("midgrade"), "Midgrade")
                           .when(fuel.contains("regular"), "Regular")
                           .when(fuel.isNull() | (fuel == ""), "Unknown")
                           .otherwise(F.initcap(fuel)))
        year_median = df.approxQuantile("vehicle_year", [0.5], 0.01)
        if year_median:
            df = df.fillna({"vehicle_year": int(round(year_median[0]))})
        for c in numeric_columns[1:]:
            mean_value = df.select(F.mean(c)).first()[0]
            if mean_value is not None:
                df = df.fillna({c: float(mean_value)})
        df = df.fillna({c: "Unknown" for c in text_columns})
        for c in ["engine_displacement_l", "city_mpg", "highway_mpg", "combined_mpg", "annual_fuel_cost_usd", "co2_grams_per_mile"]:
            df = df.withColumn(c, F.round(F.col(c), 2))
        df = df.withColumn("vehicle_year", F.col("vehicle_year").cast("int"))
        df = df.withColumn("engine_cylinders", F.round("engine_cylinders").cast("int"))
        ref_year = df.agg(F.max("vehicle_year").alias("y")).first()["y"]
        df = df.withColumn("Vehicle_Age", F.lit(ref_year) - F.col("vehicle_year"))
        df = df.withColumn("Fuel_Economy_Category", F.when(F.col("combined_mpg") < 15, "Low Efficiency").when(F.col("combined_mpg") < 25, "Medium Efficiency").otherwise("High Efficiency"))
        df = df.withColumn("Fuel_Cost_Category", F.when(F.col("annual_fuel_cost_usd") >= 3000, "High Cost").when(F.col("annual_fuel_cost_usd") >= 2000, "Medium Cost").otherwise("Low Cost"))
        df = df.withColumn("Emission_Category", F.when(F.col("co2_grams_per_mile") < 300, "Low Emission").when(F.col("co2_grams_per_mile") <= 600, "Medium Emission").otherwise("High Emission"))
        df = df.withColumn("Vehicle_Performance_Level", F.when((F.col("combined_mpg") >= 25) & (F.col("co2_grams_per_mile") < 400), "Efficient").when((F.col("combined_mpg") < 15) | (F.col("co2_grams_per_mile") > 600), "Inefficient").otherwise("Average"))
        df = df.withColumn("Powertrain_Category", F.when(F.col("fuel_type") == "Electricity / Hybrid", "Electric / Hybrid").when(F.col("fuel_type") == "Diesel", "Diesel").when(F.col("fuel_type").isin("E85 / Flex Fuel", "CNG"), "Alternative Fuel").otherwise("Gasoline"))
        df = df.withColumn("Year_Group", F.when(F.col("vehicle_year") < 2000, "Before 2000").when(F.col("vehicle_year") < 2010, "2000-2009").otherwise("2010+") )

        # Avoid Spark toPandas() because some PySpark versions import distutils,
        # which is unavailable in Python 3.12+. Collect rows directly instead.
        cleaned_pdf = pd.DataFrame([row.asDict(recursive=True) for row in df.collect()])
        missing_after = int(cleaned_pdf.isna().sum().sum())
        transformed_path = OUTPUT_DIR / "transportation_fuel_transformed.csv"
        cleaned_path = OUTPUT_DIR / "transportation_fuel_cleaned.csv"
        analysis_path = OUTPUT_DIR / "transportation_fuel_analysis.csv"
        cleaned_pdf[expected].to_csv(cleaned_path, index=False)
        cleaned_pdf.to_csv(transformed_path, index=False)
        analysis_pdf = (cleaned_pdf.groupby("fuel_type", dropna=False)
                        .agg(Vehicle_Count=("fuel_type", "size"), Average_Combined_MPG=("combined_mpg", "mean"),
                             Total_Annual_Fuel_Cost=("annual_fuel_cost_usd", "sum"), Average_Annual_Fuel_Cost=("annual_fuel_cost_usd", "mean"),
                             Average_CO2=("co2_grams_per_mile", "mean"), Minimum_CO2=("co2_grams_per_mile", "min"), Maximum_CO2=("co2_grams_per_mile", "max"))
                        .reset_index().round(2).sort_values("Vehicle_Count", ascending=False))
        analysis_pdf.to_csv(analysis_path, index=False)
        quality = {"original_rows": int(original_rows), "original_columns": int(original_columns), "duplicate_count": int(duplicate_count),
                   "cleaned_rows": int(len(cleaned_pdf)), "removed_duplicates": int(duplicate_count), "missing_before": int(missing_before),
                   "missing_after": missing_after, "invalid_year": int(invalid_year), "invalid_mpg": int(invalid_mpg),
                   "invalid_cost": int(invalid_cost), "invalid_co2": int(invalid_co2), "reference_year": int(ref_year or 0),
                   "outputs": [str(cleaned_path), str(transformed_path), str(analysis_path)]}
        return cleaned_pdf, quality
    finally:
        spark.stop()


def money(value):
    return "$" + format(float(value or 0), ",.0f")


def kpi(label, value, note):
    st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>', unsafe_allow_html=True)


def section(title, subtitle=None):
    st.markdown(f'<div class="section-title">{title}</div>' + (f'<div class="section-sub">{subtitle}</div>' if subtitle else ""), unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<div style='font-size:12px;letter-spacing:2px;color:#83e1d7;font-weight:700;margin:8px 0 8px'>F U E L S C O P E</div><div style='font-size:22px;font-weight:800;margin-bottom:4px'>Vehicle Intelligence</div><div style='font-size:12px;color:#9eafc8;margin-bottom:24px'>PySpark ETL & Analytics</div>", unsafe_allow_html=True)
    page = st.radio("WORKSPACE", ["Overview", "Fuel & Emissions", "Vehicle Explorer", "Data Quality", "Downloads"], label_visibility="visible")
    st.markdown("---")
    st.markdown("**Dataset source**")
    uploaded = st.file_uploader("Upload a replacement CSV", type=["csv"], help="Must include the 14 vehicle fuel dataset columns.")
    st.caption("Processing runs locally with PySpark. CSV exports use Pandas, not Spark/Hadoop file writes.")

if uploaded is not None:
    content = uploaded.getvalue()
    signature = hashlib.md5(content).hexdigest()
    upload_path = DATA_DIR / f"uploaded_{signature}.csv"
    if not upload_path.exists():
        upload_path.write_bytes(content)
    selected_path = upload_path
    dataset_label = uploaded.name
else:
    selected_path = DEFAULT_CSV
    dataset_label = DEFAULT_CSV.name

if not selected_path.exists():
    st.error(f"Dataset not found: {selected_path}")
    st.stop()

try:
    with st.spinner("Running PySpark cleaning and transformation pipeline..."):
        data, quality = process_dataset(str(selected_path), str(selected_path.stat().st_mtime_ns))
except Exception as exc:
    st.error("The ETL pipeline could not process this file.")
    st.exception(exc)
    st.info("Check that the CSV contains all required columns and that Java is installed. See README.txt for setup steps.")
    st.stop()

st.markdown(f"<div class='hero'><div class='eyebrow'>TRANSPORTATION DATA • ETL INTELLIGENCE</div><h1>Fuel efficiency, cost & emissions</h1><p>Turn inconsistent vehicle records into clear, decision-ready insights. Current dataset: <b>{dataset_label}</b></p></div>", unsafe_allow_html=True)

# Filters apply to all visual analysis and the explorer.
with st.container():
    f1, f2, f3 = st.columns([1.15, 1.15, 1.7])
    years = sorted([int(x) for x in data["vehicle_year"].dropna().unique()])
    year_range = f1.select_slider("Vehicle year range", options=years, value=(years[0], years[-1])) if years else (0, 9999)
    fuel_options = sorted(data["fuel_type"].fillna("Unknown").unique().tolist())
    chosen_fuels = f2.multiselect("Fuel types", fuel_options, default=fuel_options)
    class_options = sorted(data["vehicle_class"].fillna("Unknown").unique().tolist())
    chosen_classes = f3.multiselect("Vehicle classes", class_options, default=class_options)
filtered = data[(data["vehicle_year"] >= year_range[0]) & (data["vehicle_year"] <= year_range[1])]
if chosen_fuels:
    filtered = filtered[filtered["fuel_type"].isin(chosen_fuels)]
else:
    filtered = filtered.iloc[0:0]
if chosen_classes:
    filtered = filtered[filtered["vehicle_class"].isin(chosen_classes)]
else:
    filtered = filtered.iloc[0:0]

if filtered.empty:
    st.warning("No vehicles match the selected filters. Select at least one fuel type and vehicle class.")
    st.stop()

if page == "Overview":
    section("Executive overview", "A high-level snapshot of the filtered vehicle fleet")
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("VEHICLES ANALYZED", f"{len(filtered):,}", f"of {quality['original_rows']:,} source rows")
    with c2: kpi("AVG. COMBINED MPG", f"{filtered['combined_mpg'].mean():.1f}", "Miles per gallon")
    with c3: kpi("AVG. ANNUAL FUEL COST", money(filtered["annual_fuel_cost_usd"].mean()), "Per vehicle / year")
    with c4: kpi("AVG. CO₂ EMISSIONS", f"{filtered['co2_grams_per_mile'].mean():.0f}", "Grams per mile")
    st.markdown("")
    left, right = st.columns([1.15, 1])
    with left:
        section("Fuel type comparison", "Vehicle count and average fuel economy")
        agg = filtered.groupby("fuel_type", as_index=False).agg(Vehicles=("model", "size"), Average_MPG=("combined_mpg", "mean")).sort_values("Vehicles", ascending=True)
        fig = px.bar(agg, x="Vehicles", y="fuel_type", orientation="h", color="Average_MPG", color_continuous_scale="Tealgrn", labels={"fuel_type":"Fuel type", "Vehicles":"Vehicles", "Average_MPG":"Avg MPG"})
        fig.update_layout(margin=dict(l=5,r=5,t=10,b=5), height=350, paper_bgcolor="white", plot_bgcolor="white", font_color="#526178", coloraxis_colorbar_title="Avg MPG")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        section("Emission profile", "Share of vehicles by CO₂ emission category")
        emission = filtered["Emission_Category"].value_counts().rename_axis("Category").reset_index(name="Vehicles")
        fig = px.pie(emission, names="Category", values="Vehicles", hole=.62, color_discrete_sequence=["#18a999", "#f0b44d", "#e56b6f"])
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(margin=dict(l=5,r=5,t=10,b=5), height=350, paper_bgcolor="white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    left, right = st.columns(2)
    with left:
        section("Fuel cost vs. efficiency", "Each point represents a vehicle record")
        fig = px.scatter(filtered, x="combined_mpg", y="annual_fuel_cost_usd", color="fuel_type", hover_data=["make","model","vehicle_year"], opacity=.62, labels={"combined_mpg":"Combined MPG", "annual_fuel_cost_usd":"Annual fuel cost (USD)", "fuel_type":"Fuel type"})
        fig.update_layout(height=360, margin=dict(l=5,r=5,t=10,b=5), paper_bgcolor="white", plot_bgcolor="white", font_color="#526178")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        section("Vehicle class performance", "Average MPG by vehicle class")
        agg = filtered.groupby("vehicle_class", as_index=False)["combined_mpg"].mean().sort_values("combined_mpg", ascending=True).tail(12)
        fig = px.bar(agg, x="combined_mpg", y="vehicle_class", orientation="h", color="combined_mpg", color_continuous_scale="Blues", labels={"combined_mpg":"Average combined MPG", "vehicle_class":"Vehicle class"})
        fig.update_layout(height=360, margin=dict(l=5,r=5,t=10,b=5), paper_bgcolor="white", plot_bgcolor="white", font_color="#526178", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    section("Quick insights")
    best = filtered.loc[filtered["combined_mpg"].idxmax()]
    costly = filtered.loc[filtered["annual_fuel_cost_usd"].idxmax()]
    low_co2 = filtered.loc[filtered["co2_grams_per_mile"].idxmin()]
    a,b,c = st.columns(3)
    with a: st.info(f"**Highest fuel economy**\n\n{best['vehicle_year']} {best['make']} {best['model']} — {best['combined_mpg']:.1f} MPG")
    with b: st.warning(f"**Highest annual fuel cost**\n\n{costly['vehicle_year']} {costly['make']} {costly['model']} — {money(costly['annual_fuel_cost_usd'])}")
    with c: st.success(f"**Lowest CO₂ per mile**\n\n{low_co2['vehicle_year']} {low_co2['make']} {low_co2['model']} — {low_co2['co2_grams_per_mile']:.1f} g/mi")

elif page == "Fuel & Emissions":
    section("Fuel & emissions analysis", "Compare fuel categories on cost, efficiency and emissions")
    a,b = st.columns(2)
    fuel_summary = filtered.groupby("fuel_type", as_index=False).agg(Vehicles=("model","size"), Average_MPG=("combined_mpg","mean"), Average_Annual_Cost=("annual_fuel_cost_usd","mean"), Average_CO2=("co2_grams_per_mile","mean")).round(2)
    with a:
        section("Average annual cost by fuel type")
        fig=px.bar(fuel_summary.sort_values("Average_Annual_Cost"), x="Average_Annual_Cost", y="fuel_type", orientation="h", color="Average_Annual_Cost", color_continuous_scale="Sunset", labels={"Average_Annual_Cost":"Average annual cost (USD)","fuel_type":"Fuel type"})
        fig.update_layout(height=380, margin=dict(l=5,r=5,t=5,b=5),paper_bgcolor="white",plot_bgcolor="white",coloraxis_showscale=False)
        st.plotly_chart(fig,use_container_width=True)
    with b:
        section("Average CO₂ by fuel type")
        fig=px.bar(fuel_summary.sort_values("Average_CO2"), x="Average_CO2", y="fuel_type", orientation="h", color="Average_CO2", color_continuous_scale="Tealrose", labels={"Average_CO2":"Average CO₂ (g/mile)","fuel_type":"Fuel type"})
        fig.update_layout(height=380, margin=dict(l=5,r=5,t=5,b=5),paper_bgcolor="white",plot_bgcolor="white",coloraxis_showscale=False)
        st.plotly_chart(fig,use_container_width=True)
    section("Fuel category metrics")
    st.dataframe(fuel_summary.rename(columns={"fuel_type":"Fuel type","Vehicles":"Vehicle count","Average_MPG":"Avg combined MPG","Average_Annual_Cost":"Avg annual cost ($)","Average_CO2":"Avg CO₂ (g/mile)"}), use_container_width=True, hide_index=True)
    section("Efficiency and emissions relationship")
    fig=px.scatter(filtered,x="combined_mpg",y="co2_grams_per_mile",color="fuel_type",size="annual_fuel_cost_usd",size_max=18,hover_data=["make","model","vehicle_year"],labels={"combined_mpg":"Combined MPG","co2_grams_per_mile":"CO₂ (g/mile)","fuel_type":"Fuel type","annual_fuel_cost_usd":"Annual cost"})
    fig.update_layout(height=440,margin=dict(l=5,r=5,t=5,b=5),paper_bgcolor="white",plot_bgcolor="white")
    st.plotly_chart(fig,use_container_width=True)

elif page == "Vehicle Explorer":
    section("Vehicle explorer", "Search, filter and compare individual vehicle records")
    search = st.text_input("Search make, model or fuel type", placeholder="e.g. Ford, Civic, Diesel")
    table = filtered.copy()
    if search.strip():
        mask = table[["make","model","fuel_type","vehicle_class"]].astype(str).apply(lambda col: col.str.contains(search.strip(), case=False, na=False)).any(axis=1)
        table = table[mask]
    col1,col2,col3 = st.columns(3)
    sort_field = col1.selectbox("Sort records by", ["combined_mpg","annual_fuel_cost_usd","co2_grams_per_mile","vehicle_year","make"])
    sort_order = col2.selectbox("Order", ["Highest / Z-A first","Lowest / A-Z first"])
    limit = col3.selectbox("Rows to display", [25,50,100,500, "All"])
    table = table.sort_values(sort_field, ascending=sort_order == "Lowest / A-Z first")
    display = table if limit == "All" else table.head(int(limit))
    show_cols = ["vehicle_year","make","model","vehicle_class","fuel_type","combined_mpg","annual_fuel_cost_usd","co2_grams_per_mile","Fuel_Economy_Category","Emission_Category"]
    st.caption(f"Showing {len(display):,} of {len(table):,} matching vehicles")
    st.dataframe(display[show_cols].rename(columns={"vehicle_year":"Year","make":"Make","model":"Model","vehicle_class":"Vehicle class","fuel_type":"Fuel type","combined_mpg":"Combined MPG","annual_fuel_cost_usd":"Annual cost ($)","co2_grams_per_mile":"CO₂ (g/mile)","Fuel_Economy_Category":"Fuel economy","Emission_Category":"Emission category"}), use_container_width=True, hide_index=True, height=520)
    st.download_button("Download filtered vehicles CSV", table.to_csv(index=False).encode("utf-8"), file_name="filtered_vehicle_records.csv", mime="text/csv")

elif page == "Data Quality":
    section("Data quality & ETL audit", "Before-and-after checks from the PySpark pipeline")
    a,b,c,d = st.columns(4)
    with a: kpi("SOURCE ROWS", f"{quality['original_rows']:,}", f"{quality['original_columns']} expected columns")
    with b: kpi("DUPLICATES REMOVED", f"{quality['removed_duplicates']:,}", "Exact duplicate rows")
    with c: kpi("CLEANED ROWS", f"{quality['cleaned_rows']:,}", "Rows retained after de-duplication")
    with d: kpi("MISSING CELLS AFTER", f"{quality['missing_after']:,}", "Across transformed dataset")
    st.markdown("")
    issues = pd.DataFrame({"Quality check":["Missing / empty cells (before)","Exact duplicate records","Invalid vehicle years","Invalid MPG records","Negative fuel cost records","Negative CO₂ records"],"Records / cells flagged":[quality["missing_before"],quality["duplicate_count"],quality["invalid_year"],quality["invalid_mpg"],quality["invalid_cost"],quality["invalid_co2"]]})
    left,right=st.columns([1.05,1])
    with left:
        section("Issues detected in source data")
        fig=px.bar(issues.sort_values("Records / cells flagged"),x="Records / cells flagged",y="Quality check",orientation="h",color="Records / cells flagged",color_continuous_scale="Teal",text_auto=True)
        fig.update_layout(height=370,margin=dict(l=5,r=5,t=5,b=5),paper_bgcolor="white",plot_bgcolor="white",coloraxis_showscale=False)
        st.plotly_chart(fig,use_container_width=True)
    with right:
        section("ETL actions applied")
        st.markdown("""
        <div class='panel'>
        <p><b>01 · De-duplicate</b><br><span class='small-note'>Removed exact duplicate records.</span></p>
        <p><b>02 · Standardize text</b><br><span class='small-note'>Trimmed whitespace, normalized casing and consolidated fuel labels.</span></p>
        <p><b>03 · Validate numeric values</b><br><span class='small-note'>Invalid years, non-positive MPG values and negative costs/emissions were marked missing.</span></p>
        <p><b>04 · Impute missing values</b><br><span class='small-note'>Numeric fields use column means (vehicle year uses median); text fields use Unknown.</span></p>
        <p><b>05 · Enrich records</b><br><span class='small-note'>Added vehicle age, fuel economy/cost categories, emission category and performance level.</span></p>
        <p><b>06 · Export safely</b><br><span class='small-note'>CSV files are exported through Pandas. The app does not call Spark's Hadoop-backed CSV writer.</span></p>
        </div>""", unsafe_allow_html=True)
    section("Data quality counters")
    st.dataframe(issues, use_container_width=True, hide_index=True)
    st.caption("Note: a flagged count can overlap with missing values. The pipeline keeps rows and imputes invalid numeric values rather than silently dropping whole vehicles.")

else:
    section("Downloads & exports", "Export cleaned data and analysis results for reports or further work")
    st.markdown("The ETL pipeline creates three CSV outputs in the project's **outputs** folder. They are regenerated when the selected dataset changes.")
    files = [("Cleaned dataset","transportation_fuel_cleaned.csv","Original columns after cleaning and standardization."),("Transformed dataset","transportation_fuel_transformed.csv","Cleaned data plus derived analytics categories."),("Fuel analysis summary","transportation_fuel_analysis.csv","Aggregated vehicle count, MPG, fuel cost and CO₂ by fuel type.")]
    for title, filename, description in files:
        path = OUTPUT_DIR / filename
        with st.container(border=True):
            c1,c2=st.columns([4,1])
            c1.markdown(f"**{title}**  \n<small>{description}</small>",unsafe_allow_html=True)
            if path.exists():
                c2.download_button("Download CSV",path.read_bytes(),file_name=filename,mime="text/csv",key=filename)
            else:
                c2.caption("Not generated")
    st.markdown("---")
    st.markdown("**Current run summary**")
    st.json({"dataset": dataset_label, "source_rows": quality["original_rows"], "cleaned_rows": quality["cleaned_rows"], "duplicates_removed": quality["removed_duplicates"], "reference_year_for_vehicle_age": quality["reference_year"]})

st.markdown("---")
st.markdown("<div style='text-align:center;color:#8a97aa;font-size:11px'>F U E L S C O P E &nbsp; • &nbsp; PySpark ETL & Analytics &nbsp; • &nbsp; Built for transportation fuel efficiency and sustainability insights</div>", unsafe_allow_html=True)
