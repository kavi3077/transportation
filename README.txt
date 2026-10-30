F U E L S C O P E — TRANSPORTATION FUEL ETL DASHBOARD
=====================================================

WHAT THIS PROJECT DOES
----------------------
A professional Streamlit dashboard that uses PySpark to clean, standardize,
transform and analyze vehicle fuel-efficiency, annual fuel-cost and CO2 data.
It includes overview KPIs, fuel/emissions comparisons, vehicle search, data
quality audit and downloadable CSV outputs.

TECHNOLOGY
----------
- Python 3.10–3.12
- PySpark for data processing and transformations
- Streamlit for the web dashboard
- Plotly for interactive charts
- Pandas for dashboard display and CSV export

No Hadoop installation or HADOOP_HOME / winutils setup is required by this app.
IMPORTANT: Spark uses Hadoop libraries internally, but this app does not write
CSV files with Spark's Hadoop-backed file writer. It exports collected results
using Pandas to avoid the Windows winutils/HADOOP_HOME error that occurs with
Spark DataFrame.write.csv(). Java is still required by Spark.

REQUIREMENTS
------------
1. Install 64-bit Python 3.10, 3.11, or 3.12.
2. Install a supported 64-bit Java runtime (Java 17 is recommended) and ensure
   `java -version` works in Command Prompt.
3. Extract the ZIP to a normal folder, not inside the ZIP preview.
4. Double-click run.bat. It creates a virtual environment, installs packages,
   and launches the dashboard. The first run needs internet access for package
   installation and may take several minutes.
5. Streamlit will open a browser. If it does not, open:
   http://localhost:8501

MANUAL START (OPTIONAL)
-----------------------
Open Command Prompt in this folder and run:
  python -m venv .venv
  .venv\Scripts\activate
  python -m pip install -r requirements.txt
  python -m streamlit run app.py

DATA
----
The bundled source dataset is:
  data\transportation_fuel_analysis_messy.csv

To use another CSV, use “Upload a replacement CSV” in the sidebar. It must
contain these columns:
vehicle_year, make, model, vehicle_class, drive, transmission,
engine_cylinders, engine_displacement_l, fuel_type, city_mpg, highway_mpg,
combined_mpg, annual_fuel_cost_usd, co2_grams_per_mile

OUTPUTS
-------
The app creates these files under outputs\:
- transportation_fuel_cleaned.csv
- transportation_fuel_transformed.csv
- transportation_fuel_analysis.csv

The cleaned dataset removes exact duplicates, normalizes text and fuel labels,
marks invalid numeric values missing, imputes missing numeric values, fills
missing text with Unknown and adds derived category columns. Rows are retained
rather than dropped for individual invalid values.

TROUBLESHOOTING
---------------
- “Java gateway process exited”: install Java 17, restart Command Prompt, and
  confirm `java -version` works. If multiple Java versions are installed,
  set JAVA_HOME to your Java 17 installation directory.
- “Python was not found”: reinstall Python and enable “Add Python to PATH”.
- Port 8501 is busy: run `python -m streamlit run app.py --server.port 8502`.
- If you previously installed incompatible packages, delete the .venv folder
  and run run.bat again.
- This app does not require Docker, Hadoop, HDFS, winutils.exe or HADOOP_HOME.
