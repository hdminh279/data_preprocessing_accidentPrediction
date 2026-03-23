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
print("[SUCCESS] Đã load model thành công trong 1 giây!")

sample_features = Vectors.dense([0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
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