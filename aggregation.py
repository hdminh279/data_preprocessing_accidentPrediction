#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DateType, TimestampType
import pandas as pd


# In[2]:


spark = SparkSession.builder \
    .master("local[*]") \
    .appName('test') \
    .getOrCreate()


# In[4]:


df_casualty = spark.read \
    .option("header", "true") \
    .csv('data/dft-road-casualty-statistics-casualty-1979-latest-published-year.csv')


# In[5]:


df_casualty.head()


# In[6]:


columns_casualty = [
    'collision_index',
    'casualty_class',
    'sex_of_casualty',
    'age_of_casualty',
    'casualty_severity',
    'casualty_type',
]


# In[7]:


col_to_drop = []
for col in df_casualty.columns:
    if col not in columns_casualty:
        col_to_drop.append(col)


# In[8]:


df_casu_drop_col = df_casualty.drop(*col_to_drop)


# In[9]:


df_casu_drop_col.show()


# In[10]:


df_casu_drop_col.schema


# In[11]:


df_casu_drop_col.count()


# In[12]:


df_casualty_1979_2024 = df_casu_drop_col.select(
    F.col("collision_index"),
    F.col("casualty_class").cast(IntegerType()),
    F.col("sex_of_casualty").cast(IntegerType()),
    F.col("age_of_casualty").cast(IntegerType()),
    F.col("casualty_severity").cast(IntegerType()),
    F.col("casualty_type").cast(IntegerType()),
)


# In[13]:


df_casualty_1979_2024.schema


# In[14]:


df_casualty_1979_2024.write.parquet("data/casualty_1979_2024", mode='overwrite')


# In[15]:


df_collision = spark.read \
    .option("header", "true") \
    .csv('data/dft-road-casualty-statistics-collision-1979-latest-published-year.csv')


# In[16]:


df_collision.head()


# In[17]:


columns_collision = [
    'collision_index',
    'collision_severity', #
    'date',
    'day_of_week',
    'time', #
    'road_type',
    'speed_limit',
    'light_conditions',
    'weather_conditions',
    'road_surface_conditions',
    'urban_or_rural_area'
]


# In[18]:


col_coll_to_drop = []
for col in df_collision.columns:
    if col not in columns_collision:
        col_coll_to_drop.append(col)


# In[19]:


df_collision_drop_col = df_collision.drop(*col_coll_to_drop)


# In[20]:


df_collision_drop_col.show()


# In[21]:


df_collision_drop_col.schema


# In[22]:


df_collision_drop_col.count()


# In[23]:


df_collision_1979_2024 = df_collision_drop_col.select(
    F.col("collision_index"),
    F.col("collision_severity").cast(IntegerType()),
    F.to_date(F.col("date"), "dd/MM/yyyy").alias("date"),
    F.col("day_of_week").cast(IntegerType()),
    F.split(F.col("time"), ":")[0].cast(IntegerType()).alias("time"),
    F.col("road_type").cast(IntegerType()),
    F.col("speed_limit").cast(IntegerType()),
    F.col("light_conditions").cast(IntegerType()),
    F.col("weather_conditions").cast(IntegerType()),
    F.col("road_surface_conditions").cast(IntegerType()),
    F.col("urban_or_rural_area").cast(IntegerType()),
)


# In[24]:


df_collision_1979_2024.show()


# In[25]:


df_collision_1979_2024.write.parquet("data/collision_1979_2024", mode='overwrite')


# In[26]:


df_coll = spark.read.parquet("data/collision_1979_2024")
df_casu = spark.read.parquet("data/casualty_1979_2024")


# In[27]:


df_coll.count()


# In[28]:


df_casu.count()


# In[29]:


df_1979_2024 = df_casu.join(df_coll, on="collision_index", how="inner")


# In[30]:


df_1979_2024.show()


# In[32]:


df_1979_2024.count()


# In[31]:


df_1979_2024.write.parquet('data/aggregation/1979-2024', mode='overwrite')


# In[32]:


df_casualty_2025 = spark.read \
    .option("header", "true") \
    .csv('data/dft-road-casualty-statistics-casualty-provisional-2025.csv')


# In[33]:


df_casu_2025_drop_col = df_casualty_2025.drop(*col_to_drop)


# In[34]:


df_casu_2025_drop_col.show()


# In[35]:


df_casu_2025_drop_col.count()


# In[36]:


df_casualty_2025 = df_casu_2025_drop_col.select(
    F.col("collision_index"),
    F.col("casualty_class").cast(IntegerType()),
    F.col("sex_of_casualty").cast(IntegerType()),
    F.col("age_of_casualty").cast(IntegerType()),
    F.col("casualty_severity").cast(IntegerType()),
    F.col("casualty_type").cast(IntegerType()),
)


# In[37]:


df_casualty_2025.schema


# In[38]:


df_collision_2025 = spark.read \
    .option("header", "true") \
    .csv('data/dft-road-casualty-statistics-collision-provisional-2025.csv')


# In[39]:


df_coll_2025_drop_col = df_collision_2025.drop(*col_coll_to_drop)


# In[40]:


df_coll_2025_drop_col.show()


# In[41]:


df_coll_2025_drop_col.count()


# In[42]:


df_collision_2025 = df_coll_2025_drop_col.select(
    F.col("collision_index"),
    F.col("collision_severity").cast(IntegerType()),
    F.to_date(F.col("date"), "dd/MM/yyyy").alias("date"),
    F.col("day_of_week").cast(IntegerType()),
    F.split(F.col("time"), ":")[0].cast(IntegerType()).alias("time"),
    F.col("road_type").cast(IntegerType()),
    F.col("speed_limit").cast(IntegerType()),
    F.col("light_conditions").cast(IntegerType()),
    F.col("weather_conditions").cast(IntegerType()),
    F.col("road_surface_conditions").cast(IntegerType()),
    F.col("urban_or_rural_area").cast(IntegerType()),
)


# In[43]:


df_collision_2025.schema


# In[44]:


df_collision_2025.show()


# In[45]:


df_2025 = df_casualty_2025.join(df_collision_2025, on="collision_index", how="inner")


# In[46]:


df_2025.count()


# In[47]:


df_2025.write.parquet('data/aggregation/2025', mode='overwrite')


# In[48]:


df_aggregation = df_1979_2024.unionByName(df_2025)


# In[49]:


df_aggregation.write.parquet('data/aggregation/1979-2025', mode = 'overwrite')


# In[ ]:




