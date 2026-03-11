#!/usr/bin/env python
# coding: utf-8
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

spark = SparkSession.builder \
    .master("local[*]") \
    .appName("test") \
    .getOrCreate()

df_data = spark.read \
    .option("recursiveFileLookup", "true") \
    .parquet("../data/aggregation/1979-2025")

# Handle missing value
# Settings null value

null_col_set = [
    "road_type",
    "casualty_class",
    "sex_of_casualty",
    "age_of_casualty",
    "speed_limit",
    "light_conditions",
    "weather_conditions",
    "road_surface_conditions",
    "urban_or_rural_area",
]

for col in null_col_set:
    df_data = df_data.withColumn(col, F.when((F.col(col) == -1), None).otherwise(F.col(col)))

df_data = df_data \
    .withColumn("sex_of_casualty", F.when((F.col("sex_of_casualty") == -1) | (F.col("sex_of_casualty") == 9), None).otherwise(F.col("sex_of_casualty")))

# Check missing value
def check_missing_val(df):
    check_list = []

    for col_name, dtype in df.dtypes:
        condition = F.col(col_name).isNull()

        if dtype in ['double', 'float']:
            condition = condition | F.isnan(F.col(col_name))

        check_list.append(F.count(F.when(condition, col_name)).alias(col_name))
    df.select(*check_list).show(vertical=True)

# Handle missing value drop null value
df_clean = df_data.dropna(subset=['casualty_class', 'date', 'time', 'road_type'])

# Handle duplicate value
df_clean = df_clean.dropDuplicates(["collision_index","casualty_class","age_of_casualty", "sex_of_casualty"])

# Handle missing value with median for speed and age
median_speed = df_clean.approxQuantile("speed_limit", [0.5], 0.01)[0]
median_age = df_clean.approxQuantile("age_of_casualty", [0.5], 0.01)[0]

df_clean = df_clean.fillna({
    "speed_limit": median_speed,
    "age_of_casualty": median_age
})

# Handle missing value with mode for sex, light, weather and road surface
mode_cols = [
    'sex_of_casualty',
    'light_conditions',
    'weather_conditions',
    'road_surface_conditions'
]
mode_values = df_clean.agg(*[F.mode(c).alias(c) for c in mode_cols]).collect()[0].asDict()

df_clean = df_clean.fillna(mode_values)
df_clean = df_clean.fillna({"urban_or_rural_area": 3})

check_missing_val(df_clean)

# Binning col time into 4 bins
# 1: morning, 2: noon, 3: afternoon, 4: night

df_clean = df_clean.withColumn("time_bin",
        F.when((F.col("time") >=5 ) & (F.col("time") < 11), 1) # 1: Morning
        .when((F.col("time") >=11 ) & (F.col("time") < 14), 2) # 2: Noon
        .when((F.col("time") >=14 ) & (F.col("time") < 18), 3) # 3: Afternoon
        .otherwise(4)
)

# Encoded data using StringIndexer

categorical_cols = [
    "road_type",
    "casualty_class",
    "sex_of_casualty",
    "light_conditions",
    "weather_conditions",
    "road_surface_conditions",
    "casualty_type",
    "urban_or_rural_area",
    "time_bin"
]

indexers = [
    StringIndexer(inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep")
    for col in categorical_cols
]

pipeline = Pipeline(stages=indexers)
df_clean_encoded = pipeline.fit(df_clean).transform(df_clean)


# Cột nhãn
label_col = "casualty_severity"

# Các cột categorical đã được mã hóa ở bước 3
encoded_feature_cols = [
    "road_type_idx",
    "casualty_class_idx",
    "sex_of_casualty_idx",
    "light_conditions_idx",
    "weather_conditions_idx",
    "road_surface_conditions_idx",
    "urban_or_rural_area_idx",
    "time_bin_idx",
    "casualty_type_idx",
    "day_of_week"
]

# Các cột số
numeric_feature_cols = [
    "age_of_casualty",
    "speed_limit"
]

# Gộp tất cả cột feature
feature_cols = numeric_feature_cols + encoded_feature_cols

# Tránh trường hợp label bị nằm trong feature
feature_cols = [c for c in feature_cols if c != label_col]

print("Danh sách feature dùng cho mô hình:")
for c in feature_cols:
    print("-", c)


df_outlier = df_clean_encoded

# Các cột số cần loại ngoại lai
outlier_cols = [
    "age_of_casualty",
    "speed_limit"
]

# Lưu ngưỡng IQR của từng cột
outlier_bounds = {}

for col_name in outlier_cols:
    q1, q3 = df_outlier.approxQuantile(col_name, [0.25, 0.75], 0.01)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_bounds[col_name] = {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound
    }

    print(f"\nCột: {col_name}")
    print(f"Q1 = {q1}")
    print(f"Q3 = {q3}")
    print(f"IQR = {iqr}")
    print(f"Lower bound = {lower_bound}")
    print(f"Upper bound = {upper_bound}")

# Đếm số dòng trước khi lọc
before_outlier_count = df_outlier.count()
print("\nSố dòng trước khi loại ngoại lai:", before_outlier_count)

# Lọc ngoại lai lần lượt theo từng cột
df_no_outlier = df_outlier

for col_name in outlier_cols:
    lower_bound = outlier_bounds[col_name]["lower_bound"]
    upper_bound = outlier_bounds[col_name]["upper_bound"]

    df_no_outlier = df_no_outlier.filter(
        (F.col(col_name) >= lower_bound) & (F.col(col_name) <= upper_bound)
    )

# Đếm số dòng sau khi lọc
after_outlier_count = df_no_outlier.count()
print("Số dòng sau khi loại ngoại lai:", after_outlier_count)
print("Số dòng bị loại:", before_outlier_count - after_outlier_count)

# Kiểm tra lại min/max sau khi loại ngoại lai
for col_name in outlier_cols:
    print(f"\nKiểm tra lại cột: {col_name}")
    df_no_outlier.select(
        F.min(col_name).alias("min_value"),
        F.max(col_name).alias("max_value")
    ).show()

df_balance_input = df_no_outlier

# Kiểm tra phân bố lớp trước khi cân bằng
print("Phân bố lớp trước khi cân bằng:")
df_balance_input.groupBy(label_col).count().orderBy(label_col).show()

# Tính số lượng từng lớp
class_count_df = df_balance_input.groupBy(label_col).count()

# Lấy số lượng lớn nhất
max_class_count = class_count_df.agg(F.max("count")).collect()[0][0]
print("Số lượng lớp lớn nhất:", max_class_count)

# Tạo trọng số cho từng lớp
class_weight_df = class_count_df.withColumn(
    "class_weight",
    F.lit(max_class_count) / F.col("count")
)

print("Trọng số của từng lớp:")
class_weight_df.orderBy(label_col).show()

# Join trọng số vào dataframe chính
df_weighted = df_balance_input.join(
    class_weight_df.select(label_col, "class_weight"),
    on=label_col,
    how="left"
)

# Kiểm tra dữ liệu sau khi thêm class_weight
df_weighted.select(label_col, "class_weight").show(10)

df_scale_input = df_weighted

numeric_assembler = VectorAssembler(
    inputCols=numeric_feature_cols,
    outputCol="numeric_features_before_scaling",
    handleInvalid="keep"
)

df_numeric_vector = numeric_assembler.transform(df_scale_input)

numeric_scaler = StandardScaler(
    inputCol="numeric_features_before_scaling",
    outputCol="numeric_features_scaled",
    withMean=True,
    withStd=True
)

numeric_scaler_model = numeric_scaler.fit(df_numeric_vector)
df_numeric_scaled = numeric_scaler_model.transform(df_numeric_vector)

final_assembler = VectorAssembler(
    inputCols=["numeric_features_scaled"] + encoded_feature_cols,
    outputCol="features",
    handleInvalid="keep"
)

df_final = final_assembler.transform(df_numeric_scaled)

df_final.select(
    label_col,
    "class_weight",
    "numeric_features_before_scaling",
    "numeric_features_scaled",
    "features"
).show(5, truncate=False)
