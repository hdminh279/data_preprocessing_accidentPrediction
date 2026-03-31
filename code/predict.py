from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel 
from pyspark.ml.linalg import Vectors

spark = SparkSession.builder \
    .master("local[*]") \
    .appName("Predict_Accident") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

saved_model = RandomForestClassificationModel.load("../model/random_forest_model")
print("[SUCCESS] Đã load model thành công")

# Danh sách feature dùng cho mô hình:
# - age_of_casualty
# - speed_limit
# - road_type_idx
# - casualty_class_idx
# - sex_of_casualty_idx
# - light_conditions_idx
# - weather_conditions_idx
# - road_surface_conditions_idx
# - urban_or_rural_area_idx
# - time_bin_idx
# - casualty_type_idx
# - day_of_week

# sample_features = Vectors.dense([16, 60, 0, 0, 1, 0, 0, 1, 0, 0, 7, 1])
sample_features = Vectors.dense([21,30,4,1,1,1,1,1,0,0,0,4])
demo_df = spark.createDataFrame([(sample_features,)], ["features"])

print("\n[INFO] Đang phân tích mức độ nghiêm trọng...")
demo_prediction = saved_model.transform(demo_df)
result = demo_prediction.select("prediction", "probability").collect()[0]

probabilities = result["probability"].toArray()
print(f"Mô hình dự đoán: NHÃN {int(result['prediction'])}")
print(f"Xác suất Nhãn 1 (Tử vong): {probabilities[1]*100:.2f}%")
print(f"Xác suất Nhãn 2 (Nặng): {probabilities[2]*100:.2f}%")
print(f"Xác suất Nhãn 3 (Nhẹ): {probabilities[3]*100:.2f}%")

spark.stop()