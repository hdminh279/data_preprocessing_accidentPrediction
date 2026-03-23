#!/usr/bin/env python
# coding: utf-8

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors

spark = SparkSession.builder \
    .master("local[*]") \
    .appName("Train_RandomForest") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.memory.offHeap.enabled", "true") \
    .config("spark.memory.offHeap.size", "2g") \
    .getOrCreate()

print("[INFO] Đang nạp dữ liệu từ thư mục optimize...")
df_ml = spark.read.parquet("../data/clean/optimize")


print("[INFO] Đang đếm số lượng các nhãn để Undersampling...")
counts = df_ml.groupBy("casualty_severity").count().collect()
count_dict = {row['casualty_severity']: row['count'] for row in counts}


min_class_count = min(count_dict.values())
print(f"[INFO] Tự động xác định số lượng Nhãn thiểu số nhất: {min_class_count} dòng")


fractions = {label: min_class_count / count for label, count in count_dict.items()}

print("[INFO] Đang tiến hành cắt tỉa lớp đa số để tạo tỷ lệ cân bằng 1:1:1...")

df_class_1 = df_ml.filter(F.col("casualty_severity") == 1).sample(withReplacement=False, fraction=fractions.get(1, 1.0), seed=42)
df_class_2 = df_ml.filter(F.col("casualty_severity") == 2).sample(withReplacement=False, fraction=fractions.get(2, 1.0), seed=42)
df_class_3 = df_ml.filter(F.col("casualty_severity") == 3).sample(withReplacement=False, fraction=fractions.get(3, 1.0), seed=42)

df_balanced = df_class_1.union(df_class_2).union(df_class_3)

print("[INFO] Đang chia tập Train/Test...")
df_train, df_test = df_balanced.randomSplit([0.8, 0.2], seed=42)

df_train.persist(pyspark.StorageLevel.MEMORY_AND_DISK)
df_test.persist(pyspark.StorageLevel.MEMORY_AND_DISK)


print("[INFO] Đang khởi tạo mô hình Random Forest...")
rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="casualty_severity",
    # weightCol="class_weight" 
    numTrees=100,      
    maxDepth=15,      
    maxBins=64,        
    seed=42
)

print("[INFO] Bắt đầu quá trình Huấn luyện (Training)... Lần này sẽ nhanh lắm!")
rf_model = rf.fit(df_train)
print("[SUCCESS] Huấn luyện hoàn tất!")

# 6. Đánh giá Mô hình trên tập Test
print("\n[INFO] Đang làm bài Test và chấm điểm...")
predictions = rf_model.transform(df_test)

# Đánh giá bằng F1-Score
evaluator = MulticlassClassificationEvaluator(
    labelCol="casualty_severity", 
    predictionCol="prediction", 
    metricName="f1"
)
f1_score = evaluator.evaluate(predictions)

# Đánh giá bằng Accuracy
accuracy = MulticlassClassificationEvaluator(
    labelCol="casualty_severity", 
    predictionCol="prediction", 
    metricName="accuracy"
).evaluate(predictions)

print("="*40)
print(f"HIỆU SUẤT MÔ HÌNH RANDOM FOREST (SAU KHI UNDERSAMPLING):")
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
print(f"Xác suất máy tính nghĩ nó là Nhãn 1 (Nhẹ): {probabilities[1]*100:.2f}%")
print(f"Xác suất máy tính nghĩ nó là Nhãn 2 (Nặng): {probabilities[2]*100:.2f}%")
print(f"Xác suất máy tính nghĩ nó là Nhãn 3 (Tử vong): {probabilities[3]*100:.2f}%")


print("\n[INFO] Đang lưu mô hình xuống ổ cứng...")
rf_model.write().overwrite().save("../model/random_forest_model")
print("[SUCCESS] Đã lưu mô hình thành công!")

spark.stop()