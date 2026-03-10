#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import mean, median


# In[2]:


spark = SparkSession.builder \
    .master("local[*]") \
    .appName("test") \
    .getOrCreate()


# In[3]:


df_data = spark.read \
    .option("recursiveFileLookup", "true") \
    .parquet("../data/aggregation/1979-2025")


# In[4]:


df_data.show()


# In[5]:


# Handle missing value


# In[6]:


# Settings null value


# In[7]:


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


# In[8]:


for col in null_col_set:
    df_data = df_data.withColumn(col, F.when((F.col(col) == -1), None).otherwise(F.col(col)))

df_data = df_data \
    .withColumn("sex_of_casualty", F.when((F.col("sex_of_casualty") == -1) | (F.col("sex_of_casualty") == 9), None).otherwise(F.col("sex_of_casualty")))


# In[9]:


# Check missing value
def check_missing_val(df):
    check_list = []

    for col_name, dtype in df.dtypes:
        condition = F.col(col_name).isNull()

        if dtype in ['double', 'float']:
            condition = condition | F.isnan(F.col(col_name))

        check_list.append(F.count(F.when(condition, col_name)).alias(col_name))
    df.select(*check_list).show(vertical=True)


# In[10]:


check_missing_val(df_data)


# In[11]:


# Handle missing value drop null value
df_clean = df_data.dropna(subset=['casualty_class', 'date', 'time', 'road_type'])


# In[12]:


# Handle duplicate value
df_clean = df_clean.dropDuplicates(["collision_index","casualty_class","age_of_casualty", "sex_of_casualty"])


# In[ ]:


check_missing_val(df_clean)


# In[ ]:


# Handle missing value with median for speed and age
median_speed = df_clean.approxQuantile("speed_limit", [0.5], 0.01)[0]
median_age = df_clean.approxQuantile("age_of_casualty", [0.5], 0.01)[0]


# In[ ]:


df_clean = df_clean.fillna({
    "speed_limit": median_speed,
    "age_of_casualty": median_age
})


# In[ ]:


# Handle missing value with mode for sex, light, weather and road surface
mode_cols = [
    'sex_of_casualty',
    'light_conditions',
    'weather_conditions',
    'road_surface_conditions'
]


# In[ ]:


mode_values = df_clean.agg(*[F.mode(c).alias(c) for c in mode_cols]).collect()[0].asDict()


# In[ ]:


df_clean = df_clean.fillna(mode_values)


# In[ ]:


check_missing_val(df_clean)


# In[ ]:


df_clean = df_clean.fillna({"urban_or_rural_area": 3})


# In[ ]:


check_missing_val(df_clean)


# In[ ]:


df_clean.select("time").show()


# In[ ]:


# Binning col time into 4 bins
# 1: morning, 2: noon, 3: afternoon, 4: night

df_clean = df_clean.withColumn("time_bin",
        F.when((F.col("time") >=5 ) & (F.col("time") < 11), 1) # 1: Morning
        .when((F.col("time") >=11 ) & (F.col("time") < 14), 2) # 2: Noon
        .when((F.col("time") >=14 ) & (F.col("time") < 18), 3) # 3: Afternoon
        .otherwise(4)
)


# In[ ]:


df_clean.select("time_bin").show()


# In[ ]:




