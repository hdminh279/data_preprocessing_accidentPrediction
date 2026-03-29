#!/usr/bin/env python
# coding: utf-8

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark import StorageLevel

spark = SparkSession.builder \
    .master("local[*]") \
    .appName("Train_RandomForest_FullData") \
    .config("spark.driver.memory", "8g") \
    .config("spark.memory.offHeap.enabled", "true") \
    .config("spark.memory.offHeap.size", "1g") \
    .getOrCreate()

print("[INFO] Đang nạp FULL dữ liệu từ thư mục optimize...")
df_ml = spark.read \
    .option("RecursiveFileLookup", "true") \
    .parquet("../data/clean/optimize/")

print("[INFO] Đang chia tập Train/Test bằng kỹ thuật Anti Join...")

df_with_id = df_ml.withColumn("id", F.monotonically_increasing_id())

ratio = {1: 0.8, 2: 0.8, 3: 0.8}
df_train = df_with_id.sampleBy("casualty_severity", fractions=ratio, seed=42)

df_test = df_with_id.join(df_train, on="id", how="left_anti")

df_train = df_train.drop("id")
df_test = df_test.drop("id")

df_train.persist(StorageLevel.MEMORY_AND_DISK)
df_test.persist(StorageLevel.MEMORY_AND_DISK)

print("[INFO] Đang khởi tạo mô hình Random Forest (Sử dụng class_weight)...")
rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="casualty_severity",
    weightCol="class_weight",
    numTrees=50,               
    maxDepth=10,               
    maxBins=64,        
    seed=42
)

print("[INFO] Bắt đầu quá trình Huấn luyện (Máy sẽ chạy hết công suất, đừng mở Chrome nhé!)...")
rf_model = rf.fit(df_train)
print("[SUCCESS] Huấn luyện hoàn tất!")

print("\n[INFO] Đang làm bài Test và chấm điểm...")
predictions = rf_model.transform(df_test)

evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol="casualty_severity", predictionCol="prediction", metricName="f1"
)
f1_score = evaluator_f1.evaluate(predictions)

evaluator_acc = MulticlassClassificationEvaluator(
    labelCol="casualty_severity", predictionCol="prediction", metricName="accuracy"
)
accuracy = evaluator_acc.evaluate(predictions)

print("="*40)
print(f"HIỆU SUẤT MÔ HÌNH RANDOM FOREST (FULL DATA + CLASS WEIGHT):")
print(f"- Độ chính xác tổng thể (Accuracy): {accuracy * 100:.2f}%")
print(f"- Điểm F1-Score: {f1_score * 100:.2f}%")
print("="*40)

print("\n[DEMO] THỬ NGHIỆM DỰ ĐOÁN MỘT VỤ TAI NẠN MỚI")
sample_features = Vectors.dense([1.5, 2.0, 1.0, 2.0, 1.0, 1.0, 2.0, 4.0, 0.0, 5.0, 2.0, 3.0])

demo_df = spark.createDataFrame([(sample_features,)], ["features"])
demo_prediction = rf_model.transform(demo_df)

result = demo_prediction.select("prediction", "probability").collect()[0]
predicted_label = int(result["prediction"])
probabilities = result["probability"].toArray()

print(f"Đầu vào (Features): {sample_features}")
print(f"Mô hình dự đoán Mức độ nghiêm trọng là: NHÃN {predicted_label}")
print(f"Xác suất máy tính nghĩ nó là Nhãn 1 (Tử vong): {probabilities[1]*100:.2f}%")
print(f"Xác suất máy tính nghĩ nó là Nhãn 2 (Nặng): {probabilities[2]*100:.2f}%")
print(f"Xác suất máy tính nghĩ nó là Nhãn 3 (Nhẹ): {probabilities[3]*100:.2f}%")

print("\n[INFO] Đang lưu mô hình xuống ổ cứng...")
rf_model.write().overwrite().save("../model/random_forest_model_full")
print("[SUCCESS] Đã lưu mô hình thành công!")

spark.stop()