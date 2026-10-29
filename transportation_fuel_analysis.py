from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    trim,
    lower,
    initcap,
    when,
    mean,
    count,
    sum as spark_sum,
    avg,
    min as spark_min,
    max as spark_max,
    round as spark_round,
    lit
)

spark = SparkSession.builder \
    .appName("Transportation Fuel ETL Pipeline") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

df = spark.read.csv(
    "transportation_fuel_analysis_messy.csv",
    header=True,
    inferSchema=True
)

original_rows = df.count()
original_columns = len(df.columns)

print("\nDATASET LOADED")
print("Rows:", original_rows)
print("Columns:", original_columns)

print("\nFIRST 10 RECORDS")
df.show(10, truncate=False)

print("\nSCHEMA")
df.printSchema()

print("\nSELECTED COLUMNS")
df.select(
    "make",
    "model",
    "vehicle_class",
    "fuel_type",
    "combined_mpg",
    "annual_fuel_cost_usd",
    "co2_grams_per_mile"
).show(10, truncate=False)

print("\nVEHICLES WITH LOW FUEL EFFICIENCY")
df.filter(
    col("combined_mpg") < 15
).select(
    "make",
    "model",
    "vehicle_year",
    "fuel_type",
    "combined_mpg"
).show(10, truncate=False)

print("\nVEHICLES WITH HIGH ANNUAL FUEL COST")
df.filter(
    col("annual_fuel_cost_usd") > 3000
).select(
    "make",
    "model",
    "fuel_type",
    "annual_fuel_cost_usd"
).show(10, truncate=False)

print("\nSTATISTICAL SUMMARY")
df.describe().show()

print("\nBASIC STATISTICS")
df.select(
    spark_min("vehicle_year").alias("Minimum_Year"),
    spark_max("vehicle_year").alias("Maximum_Year"),
    spark_round(
        avg("vehicle_year"),
        2
    ).alias("Average_Year"),
    spark_min(
        "combined_mpg"
    ).alias("Minimum_Combined_MPG"),
    spark_max(
        "combined_mpg"
    ).alias("Maximum_Combined_MPG"),
    spark_round(
        avg("combined_mpg"),
        2
    ).alias("Average_Combined_MPG"),
    spark_min(
        "annual_fuel_cost_usd"
    ).alias("Minimum_Fuel_Cost"),
    spark_max(
        "annual_fuel_cost_usd"
    ).alias("Maximum_Fuel_Cost"),
    spark_round(
        avg("annual_fuel_cost_usd"),
        2
    ).alias("Average_Fuel_Cost"),
    spark_min(
        "co2_grams_per_mile"
    ).alias("Minimum_CO2"),
    spark_max(
        "co2_grams_per_mile"
    ).alias("Maximum_CO2"),
    spark_round(
        avg("co2_grams_per_mile"),
        2
    ).alias("Average_CO2")
).show()

print("\nMISSING AND EMPTY VALUES")

df.select([
    spark_sum(
        when(
            col(c).isNull() |
            (trim(col(c).cast("string")) == ""),
            1
        ).otherwise(0)
    ).alias(c)
    for c in df.columns
]).show(truncate=False)

duplicate_count = (
    df.count() -
    df.dropDuplicates().count()
)

invalid_year_before = df.filter(
    (col("vehicle_year") < 1900) |
    (col("vehicle_year") > 2025)
).count()

invalid_engine_before = df.filter(
    col("engine_cylinders") < 0
).count()

invalid_mpg_before = df.filter(
    (col("city_mpg") <= 0) |
    (col("highway_mpg") <= 0) |
    (col("combined_mpg") <= 0)
).count()

negative_fuel_cost_before = df.filter(
    col("annual_fuel_cost_usd") < 0
).count()

negative_co2_before = df.filter(
    col("co2_grams_per_mile") < 0
).count()

print("\nDATA QUALITY ISSUES")
print("Duplicate Records:", duplicate_count)
print("Invalid Vehicle Years:", invalid_year_before)
print("Invalid Engine Values:", invalid_engine_before)
print("Invalid MPG Records:", invalid_mpg_before)
print("Negative Fuel Cost Records:", negative_fuel_cost_before)
print("Negative CO2 Records:", negative_co2_before)

df = df.dropDuplicates()

for c in df.columns:
    df = df.withColumn(
        c,
        when(
            trim(
                col(c).cast("string")
            ) == "",
            None
        ).otherwise(
            col(c)
        )
    )

text_columns = [
    "make",
    "model",
    "vehicle_class",
    "drive",
    "transmission",
    "fuel_type"
]

for c in text_columns:
    df = df.withColumn(
        c,
        trim(col(c))
    )

for c in [
    "make",
    "model",
    "vehicle_class",
    "drive",
    "transmission"
]:
    df = df.withColumn(
        c,
        initcap(
            lower(col(c))
        )
    )

df = df.withColumn(
    "fuel_type",
    lower(
        trim(
            col("fuel_type")
        )
    )
)

df = df.withColumn(
    "fuel_type",
    when(
        col("fuel_type").contains("electric"),
        "Electricity / Hybrid"
    )
    .when(
        col("fuel_type").contains("hybrid"),
        "Electricity / Hybrid"
    )
    .when(
        col("fuel_type").contains("e85"),
        "E85 / Flex Fuel"
    )
    .when(
        col("fuel_type").contains("flex"),
        "E85 / Flex Fuel"
    )
    .when(
        col("fuel_type").contains("diesel"),
        "Diesel"
    )
    .when(
        col("fuel_type").contains("cng"),
        "CNG"
    )
    .when(
        col("fuel_type").contains("premium"),
        "Premium"
    )
    .when(
        col("fuel_type").contains("midgrade"),
        "Midgrade"
    )
    .when(
        col("fuel_type").contains("regular"),
        "Regular"
    )
    .otherwise(
        initcap(
            col("fuel_type")
        )
    )
)

df = df.withColumn(
    "vehicle_year",
    col("vehicle_year").cast("int")
)

df = df.withColumn(
    "engine_cylinders",
    col("engine_cylinders").cast("double")
)

df = df.withColumn(
    "engine_displacement_l",
    col("engine_displacement_l").cast("double")
)

df = df.withColumn(
    "city_mpg",
    col("city_mpg").cast("double")
)

df = df.withColumn(
    "highway_mpg",
    col("highway_mpg").cast("double")
)

df = df.withColumn(
    "combined_mpg",
    col("combined_mpg").cast("double")
)

df = df.withColumn(
    "annual_fuel_cost_usd",
    col("annual_fuel_cost_usd").cast("double")
)

df = df.withColumn(
    "co2_grams_per_mile",
    col("co2_grams_per_mile").cast("double")
)

df = df.withColumn(
    "vehicle_year",
    when(
        (col("vehicle_year") < 1900) |
        (col("vehicle_year") > 2025),
        None
    ).otherwise(
        col("vehicle_year")
    )
)

df = df.withColumn(
    "engine_cylinders",
    when(
        col("engine_cylinders") < 0,
        None
    ).otherwise(
        col("engine_cylinders")
    )
)

df = df.withColumn(
    "engine_displacement_l",
    when(
        col("engine_displacement_l") < 0,
        None
    ).otherwise(
        col("engine_displacement_l")
    )
)

df = df.withColumn(
    "city_mpg",
    when(
        col("city_mpg") <= 0,
        None
    ).otherwise(
        col("city_mpg")
    )
)

df = df.withColumn(
    "highway_mpg",
    when(
        col("highway_mpg") <= 0,
        None
    ).otherwise(
        col("highway_mpg")
    )
)

df = df.withColumn(
    "combined_mpg",
    when(
        col("combined_mpg") <= 0,
        None
    ).otherwise(
        col("combined_mpg")
    )
)

df = df.withColumn(
    "annual_fuel_cost_usd",
    when(
        col("annual_fuel_cost_usd") < 0,
        None
    ).otherwise(
        col("annual_fuel_cost_usd")
    )
)

df = df.withColumn(
    "co2_grams_per_mile",
    when(
        col("co2_grams_per_mile") < 0,
        None
    ).otherwise(
        col("co2_grams_per_mile")
    )
)

average_engine_cylinders = df.select(
    mean("engine_cylinders")
).first()[0]

if average_engine_cylinders is not None:
    df = df.fillna({
        "engine_cylinders":
        float(average_engine_cylinders)
    })

average_engine_size = df.select(
    mean("engine_displacement_l")
).first()[0]

if average_engine_size is not None:
    df = df.fillna({
        "engine_displacement_l":
        float(average_engine_size)
    })

average_city_mpg = df.select(
    mean("city_mpg")
).first()[0]

if average_city_mpg is not None:
    df = df.fillna({
        "city_mpg":
        float(average_city_mpg)
    })

average_highway_mpg = df.select(
    mean("highway_mpg")
).first()[0]

if average_highway_mpg is not None:
    df = df.fillna({
        "highway_mpg":
        float(average_highway_mpg)
    })

average_combined_mpg = df.select(
    mean("combined_mpg")
).first()[0]

if average_combined_mpg is not None:
    df = df.fillna({
        "combined_mpg":
        float(average_combined_mpg)
    })

average_co2 = df.select(
    mean("co2_grams_per_mile")
).first()[0]

if average_co2 is not None:
    df = df.fillna({
        "co2_grams_per_mile":
        float(average_co2)
    })

average_fuel_cost = df.select(
    mean("annual_fuel_cost_usd")
).first()[0]

if average_fuel_cost is not None:
    df = df.fillna({
        "annual_fuel_cost_usd":
        float(average_fuel_cost)
    })

df = df.fillna({
    "make": "Unknown",
    "model": "Unknown",
    "vehicle_class": "Unknown",
    "drive": "Unknown",
    "transmission": "Unknown",
    "fuel_type": "Unknown"
})

df = df.withColumn(
    "engine_cylinders",
    spark_round(
        col("engine_cylinders"),
        0
    ).cast("int")
)

df = df.withColumn(
    "engine_displacement_l",
    spark_round(
        col("engine_displacement_l"),
        2
    )
)

df = df.withColumn(
    "city_mpg",
    spark_round(
        col("city_mpg"),
        2
    )
)

df = df.withColumn(
    "highway_mpg",
    spark_round(
        col("highway_mpg"),
        2
    )
)

df = df.withColumn(
    "combined_mpg",
    spark_round(
        col("combined_mpg"),
        2
    )
)

df = df.withColumn(
    "annual_fuel_cost_usd",
    spark_round(
        col("annual_fuel_cost_usd"),
        2
    )
)

df = df.withColumn(
    "co2_grams_per_mile",
    spark_round(
        col("co2_grams_per_mile"),
        2
    )
)

quantiles = df.approxQuantile(
    "annual_fuel_cost_usd",
    [0.25, 0.75],
    0.01
)

q1 = quantiles[0]
q3 = quantiles[1]
iqr = q3 - q1

lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)

before_outliers = df.count()

df = df.filter(
    (col("annual_fuel_cost_usd") >= lower_bound) &
    (col("annual_fuel_cost_usd") <= upper_bound)
)

after_outliers = df.count()

cleaned_rows = df.count()

print("\nCLEANING RESULTS")
print("Rows Before Cleaning:", original_rows)
print("Rows After Cleaning:", cleaned_rows)
print("Duplicates Removed:", duplicate_count)
print(
    "Outliers Removed:",
    before_outliers - after_outliers
)

print("\nMISSING VALUES AFTER CLEANING")

df.select([
    spark_sum(
        when(
            col(c).isNull() |
            (
                trim(
                    col(c).cast("string")
                ) == ""
            ),
            1
        ).otherwise(0)
    ).alias(c)
    for c in df.columns
]).show(truncate=False)

df.toPandas().to_csv(
    "transportation_fuel_cleaned.csv",
    index=False
)

print(
    "\nCleaned dataset saved: "
    "transportation_fuel_cleaned.csv"
)

reference_year = df.select(
    spark_max("vehicle_year")
).first()[0]

df = df.withColumn(
    "Vehicle_Age",
    lit(reference_year) -
    col("vehicle_year")
)

df = df.withColumn(
    "Fuel_Economy_Category",
    when(
        col("combined_mpg") < 15,
        "Low Efficiency"
    )
    .when(
        col("combined_mpg") < 25,
        "Medium Efficiency"
    )
    .otherwise(
        "High Efficiency"
    )
)

df = df.withColumn(
    "Fuel_Cost_Category",
    when(
        col("annual_fuel_cost_usd") >= 3000,
        "High Cost"
    )
    .when(
        col("annual_fuel_cost_usd") >= 2000,
        "Medium Cost"
    )
    .otherwise(
        "Low Cost"
    )
)

df = df.withColumn(
    "Emission_Category",
    when(
        col("co2_grams_per_mile") < 300,
        "Low Emission"
    )
    .when(
        col("co2_grams_per_mile") <= 600,
        "Medium Emission"
    )
    .otherwise(
        "High Emission"
    )
)

df = df.withColumn(
    "Vehicle_Performance_Level",
    when(
        (col("combined_mpg") >= 25) &
        (col("co2_grams_per_mile") < 400),
        "Efficient"
    )
    .when(
        (col("combined_mpg") < 15) |
        (col("co2_grams_per_mile") > 600),
        "Inefficient"
    )
    .otherwise(
        "Average"
    )
)

df = df.withColumn(
    "Powertrain_Category",
    when(
        col("fuel_type") ==
        "Electricity / Hybrid",
        "Electric / Hybrid"
    )
    .when(
        col("fuel_type") == "Diesel",
        "Diesel"
    )
    .when(
        col("fuel_type").isin(
            "E85 / Flex Fuel",
            "CNG"
        ),
        "Alternative Fuel"
    )
    .otherwise(
        "Gasoline"
    )
)

df = df.withColumn(
    "Year_Group",
    when(
        col("vehicle_year") < 2000,
        "Before 2000"
    )
    .when(
        col("vehicle_year") < 2010,
        "2000-2009"
    )
    .otherwise(
        "2010-2017"
    )
)

print("\nTRANSFORMED DATA")

df.select(
    "vehicle_year",
    "make",
    "model",
    "fuel_type",
    "combined_mpg",
    "annual_fuel_cost_usd",
    "co2_grams_per_mile",
    "Vehicle_Age",
    "Fuel_Economy_Category",
    "Fuel_Cost_Category",
    "Emission_Category",
    "Vehicle_Performance_Level"
).show(
    20,
    truncate=False
)

print("\nFUEL ECONOMY CATEGORY VALIDATION")

df.groupBy(
    "Fuel_Economy_Category"
).count().orderBy(
    col("count").desc()
).show()

print("\nFUEL COST CATEGORY VALIDATION")

df.groupBy(
    "Fuel_Cost_Category"
).count().orderBy(
    col("count").desc()
).show()

print("\nEMISSION CATEGORY VALIDATION")

df.groupBy(
    "Emission_Category"
).count().orderBy(
    col("count").desc()
).show()

print("\nVEHICLE PERFORMANCE VALIDATION")

df.groupBy(
    "Vehicle_Performance_Level"
).count().orderBy(
    col("count").desc()
).show()

print("\nPOWERTRAIN CATEGORY VALIDATION")

df.groupBy(
    "Powertrain_Category"
).count().orderBy(
    col("count").desc()
).show()

invalid_mpg = df.filter(
    (col("city_mpg") <= 0) |
    (col("highway_mpg") <= 0) |
    (col("combined_mpg") <= 0)
).count()

invalid_fuel_cost = df.filter(
    col("annual_fuel_cost_usd") < 0
).count()

invalid_co2 = df.filter(
    col("co2_grams_per_mile") < 0
).count()

print(
    "\nInvalid MPG Records:",
    invalid_mpg
)

print(
    "Invalid Fuel Cost Records:",
    invalid_fuel_cost
)

print(
    "Invalid CO2 Records:",
    invalid_co2
)

df.toPandas().to_csv(
    "transportation_fuel_transformed.csv",
    index=False
)

print(
    "\nTransformed dataset saved: "
    "transportation_fuel_transformed.csv"
)

print("\nVEHICLE COUNT BY FUEL TYPE")

df.groupBy(
    "fuel_type"
).count().orderBy(
    col("count").desc()
).show()

print("\nVEHICLE COUNT BY VEHICLE CLASS")

df.groupBy(
    "vehicle_class"
).count().orderBy(
    col("count").desc()
).show()

print("\nVEHICLE COUNT BY DRIVE TYPE")

df.groupBy(
    "drive"
).count().orderBy(
    col("count").desc()
).show()

print("\nFUEL TYPE ANALYSIS")

df.groupBy(
    "fuel_type"
).agg(
    count("*").alias(
        "Vehicle_Count"
    ),
    spark_round(
        avg("combined_mpg"),
        2
    ).alias(
        "Average_Combined_MPG"
    ),
    spark_round(
        avg("annual_fuel_cost_usd"),
        2
    ).alias(
        "Average_Annual_Fuel_Cost"
    ),
    spark_round(
        avg("co2_grams_per_mile"),
        2
    ).alias(
        "Average_CO2"
    )
).orderBy(
    col("Vehicle_Count").desc()
).show(
    truncate=False
)

print("\nVEHICLE CLASS ANALYSIS")

df.groupBy(
    "vehicle_class"
).agg(
    count("*").alias(
        "Vehicle_Count"
    ),
    spark_round(
        avg("combined_mpg"),
        2
    ).alias(
        "Average_Combined_MPG"
    ),
    spark_round(
        avg("annual_fuel_cost_usd"),
        2
    ).alias(
        "Average_Annual_Fuel_Cost"
    ),
    spark_round(
        avg("co2_grams_per_mile"),
        2
    ).alias(
        "Average_CO2"
    )
).orderBy(
    col("Vehicle_Count").desc()
).show(
    truncate=False
)

print("\nDRIVE TYPE ANALYSIS")

df.groupBy(
    "drive"
).agg(
    count("*").alias(
        "Vehicle_Count"
    ),
    spark_round(
        avg("combined_mpg"),
        2
    ).alias(
        "Average_Combined_MPG"
    ),
    spark_round(
        avg("annual_fuel_cost_usd"),
        2
    ).alias(
        "Average_Annual_Fuel_Cost"
    ),
    spark_round(
        avg("co2_grams_per_mile"),
        2
    ).alias(
        "Average_CO2"
    )
).orderBy(
    col("Vehicle_Count").desc()
).show(
    truncate=False
)

print("\nYEAR GROUP ANALYSIS")

df.groupBy(
    "Year_Group"
).agg(
    count("*").alias(
        "Vehicle_Count"
    ),
    spark_round(
        avg("combined_mpg"),
        2
    ).alias(
        "Average_Combined_MPG"
    ),
    spark_round(
        avg("annual_fuel_cost_usd"),
        2
    ).alias(
        "Average_Annual_Fuel_Cost"
    ),
    spark_round(
        avg("co2_grams_per_mile"),
        2
    ).alias(
        "Average_CO2"
    )
).orderBy(
    "Year_Group"
).show(
    truncate=False
)

print("\nEMISSION CATEGORY ANALYSIS")

df.groupBy(
    "Emission_Category"
).agg(
    count("*").alias(
        "Vehicle_Count"
    ),
    spark_round(
        avg("combined_mpg"),
        2
    ).alias(
        "Average_Combined_MPG"
    ),
    spark_round(
        avg("annual_fuel_cost_usd"),
        2
    ).alias(
        "Average_Annual_Fuel_Cost"
    ),
    spark_round(
        avg("co2_grams_per_mile"),
        2
    ).alias(
        "Average_CO2"
    )
).orderBy(
    col("Vehicle_Count").desc()
).show(
    truncate=False
)

print("\nTOP 10 VEHICLE MAKES")

df.groupBy(
    "make"
).count().orderBy(
    col("count").desc()
).show(
    10,
    truncate=False
)

print("\nTOP 10 MOST EFFICIENT VEHICLES")

df.select(
    "make",
    "model",
    "vehicle_year",
    "fuel_type",
    "combined_mpg",
    "co2_grams_per_mile"
).orderBy(
    col("combined_mpg").desc(),
    col("co2_grams_per_mile").asc()
).show(
    10,
    truncate=False
)

print("\nLOW EFFICIENCY VEHICLES")

low_efficiency = df.filter(
    col("Fuel_Economy_Category") ==
    "Low Efficiency"
)

print(
    "Low Efficiency Vehicles:",
    low_efficiency.count()
)

low_efficiency.groupBy(
    "vehicle_class"
).count().orderBy(
    col("count").desc()
).show(
    truncate=False
)

print("\nHIGH FUEL COST VEHICLES")

high_fuel_cost = df.filter(
    col("Fuel_Cost_Category") ==
    "High Cost"
)

print(
    "High Fuel Cost Vehicles:",
    high_fuel_cost.count()
)

high_fuel_cost.groupBy(
    "fuel_type"
).count().orderBy(
    col("count").desc()
).show(
    truncate=False
)

print("\nTOP 10 HIGHEST FUEL COST VEHICLES")

df.select(
    "make",
    "model",
    "vehicle_year",
    "fuel_type",
    "annual_fuel_cost_usd",
    "combined_mpg"
).orderBy(
    col("annual_fuel_cost_usd").desc()
).show(
    10,
    truncate=False
)

print("\nLOWEST EMISSION VEHICLES")

df.select(
    "make",
    "model",
    "vehicle_year",
    "fuel_type",
    "combined_mpg",
    "co2_grams_per_mile"
).orderBy(
    col("co2_grams_per_mile").asc()
).show(
    10,
    truncate=False
)

analysis_df = df.groupBy(
    "fuel_type"
).agg(
    count("*").alias(
        "Vehicle_Count"
    ),
    spark_round(
        avg("combined_mpg"),
        2
    ).alias(
        "Average_Combined_MPG"
    ),
    spark_round(
        spark_sum("annual_fuel_cost_usd"),
        2
    ).alias(
        "Total_Annual_Fuel_Cost"
    ),
    spark_round(
        avg("annual_fuel_cost_usd"),
        2
    ).alias(
        "Average_Annual_Fuel_Cost"
    ),
    spark_round(
        avg("co2_grams_per_mile"),
        2
    ).alias(
        "Average_CO2"
    ),
    spark_min(
        "co2_grams_per_mile"
    ).alias(
        "Minimum_CO2"
    ),
    spark_max(
        "co2_grams_per_mile"
    ).alias(
        "Maximum_CO2"
    )
).orderBy(
    col("Vehicle_Count").desc()
)

analysis_df.toPandas().to_csv(
    "transportation_fuel_analysis.csv",
    index=False
)

print(
    "\nAnalysis dataset saved: "
    "transportation_fuel_analysis.csv"
)

final_duplicates = (
    df.count() -
    df.dropDuplicates().count()
)

missing_final = 0

for c in df.columns:
    missing_final += df.filter(
        col(c).isNull() |
        (
            trim(
                col(c).cast("string")
            ) == ""
        )
    ).count()

print("\nFINAL VALIDATION")

print(
    "Rows:",
    df.count()
)

print(
    "Columns:",
    len(df.columns)
)

print(
    "Duplicate Records:",
    final_duplicates
)

print(
    "Missing / Empty Values:",
    missing_final
)

print(
    "Invalid MPG Records:",
    invalid_mpg
)

print(
    "Invalid Fuel Cost Records:",
    invalid_fuel_cost
)

print(
    "Invalid CO2 Records:",
    invalid_co2
)

print("\nFINAL SCHEMA")

df.printSchema()

print("\nFINAL 10 RECORDS")

df.show(
    10,
    truncate=False
)

df.toPandas().to_csv(
    "transportation_fuel_final.csv",
    index=False
)

print(
    "\nFinal dataset saved: "
    "transportation_fuel_final.csv"
)

print(
    "\nETL PIPELINE COMPLETED SUCCESSFULLY"
)

spark.stop()