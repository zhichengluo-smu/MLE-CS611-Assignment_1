#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col ,when
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table


# ## set up pyspark session

# In[2]:


# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")


# ## set up config

# In[3]:


# set up config
snapshot_date_str = "2023-01-01"

start_date_str = "2023-01-01"
end_date_str = "2024-12-01"


# In[4]:


# generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))

        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
dates_str_lst


# ## Build Features clickstream Bronze Table

# In[5]:


# create bronze datalake
bronze_clickstream_directory = "datamart/bronze/features_clickstream/"

if not os.path.exists(bronze_clickstream_directory):
    os.makedirs(bronze_clickstream_directory)


# In[6]:


# run bronze backfill
for date_str in dates_str_lst:
    utils.data_processing_bronze_table.process_bronze_table_features_clickstream(date_str, bronze_clickstream_directory, spark)


# ## Build Features Attributes Bronze Table

# In[7]:


# create bronze datalake
bronze_attributes_directory = "datamart/bronze/features_attributes/"

if not os.path.exists(bronze_attributes_directory):
    os.makedirs(bronze_attributes_directory)


# In[8]:


# run bronze backfill
for date_str in dates_str_lst:
    utils.data_processing_bronze_table.process_bronze_table_features_attributes(date_str, bronze_attributes_directory, spark)


# In[9]:


# inspect output
utils.data_processing_bronze_table.process_bronze_table_features_attributes(date_str, bronze_attributes_directory, spark).toPandas()


# ## Build Features Financial Bronze Table

# In[10]:


# create bronze datalake
bronze_financials_directory = "datamart/bronze/features_financials/"

if not os.path.exists(bronze_financials_directory):
    os.makedirs(bronze_financials_directory)


# In[11]:


# run bronze backfill
for date_str in dates_str_lst:
    utils.data_processing_bronze_table.process_bronze_table_features_financials(date_str, bronze_financials_directory, spark)


# In[12]:


# inspect output
utils.data_processing_bronze_table.process_bronze_table_features_financials(date_str, bronze_financials_directory, spark).toPandas()


# ## Build Silver Table Part 1 : attributes and financials

# In[13]:


# create bronze datalake
silver_features_directory = "datamart/silver/features/"

if not os.path.exists(silver_features_directory):
    os.makedirs(silver_features_directory)


# In[14]:


# run silver backfill
for date_str in dates_str_lst:
    utils.data_processing_silver_table.process_silver_features_table(date_str,bronze_attributes_directory, bronze_financials_directory, silver_features_directory, spark)


# In[15]:


utils.data_processing_silver_table.process_silver_features_table(date_str,bronze_attributes_directory, bronze_financials_directory, silver_features_directory, spark).toPandas()


# ## Build Silver Table Part 2 : clickstream

# In[16]:


# create bronze datalake
silver_features_clickstream_directory = "datamart/silver/features-clickstream/"

if not os.path.exists(silver_features_clickstream_directory):
    os.makedirs(silver_features_clickstream_directory)


# In[17]:


# run silver backfill
for date_str in dates_str_lst:
    utils.data_processing_silver_table.process_silver_features_clickstream_table(date_str, bronze_clickstream_directory, silver_features_clickstream_directory, spark)


# In[18]:


utils.data_processing_silver_table.process_silver_features_clickstream_table(date_str, bronze_clickstream_directory, silver_features_clickstream_directory, spark).toPandas()


# ## EDA on features

# In[19]:


# Path to the folder containing CSV files
folder_path = silver_features_directory

# Read all CSV files into a single DataFrame
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
df = spark.read.option("header", "true").parquet(*files_list)



# ## EDA : Continuous variables - Summary statistics

# In[20]:


df.select(
  "Age","Annual_Income","Outstanding_Debt","new_loan_to_income_ratio",
  "new_credit_history_months"
).describe().show()


# ## EDA : Histograms  - Summary statistics ( Annual_Income , Outstanding_Debt , Monthly_Balance )

# In[21]:


import matplotlib.pyplot as plt

pandas_df = df.select("Annual_Income", "Outstanding_Debt", "Monthly_Balance").toPandas()
# assume pandas_df is your pandas DataFrame
pandas_df[["Annual_Income","Outstanding_Debt","Monthly_Balance"]] \
    .hist(bins=30, layout=(1, 3), figsize=(15, 4), grid=False)

plt.tight_layout()
plt.show()


# ## EDA : Box‐plots of debt by credit mix

# In[22]:


# Convert Spark to pandas
df_pd = df.select("Outstanding_Debt", "new_Credit_Mix_code").toPandas()

# Plot the boxplot
import matplotlib.pyplot as plt

df_pd.boxplot(column="Outstanding_Debt", by="new_Credit_Mix_code")
plt.title("Debt by Credit Mix")
plt.suptitle("")  # remove automatic suptitle
plt.xlabel("Credit Mix Code")
plt.ylabel("Outstanding Debt")
plt.show()


# ## EDA : Behavioral Risk Profiling

# In[23]:


df.groupBy("new_Payment_Behaviour_code") \
  .agg({"Annual_Income": "avg", "Outstanding_Debt": "avg"}) \
  .orderBy("new_Payment_Behaviour_code") \
  .show()


# ## EDA : Age Bucket vs Credit History

# In[24]:


df.withColumn("age_bucket", 
    when(col("Age") < 25, "<25")
   .when(col("Age") < 40, "25-39")
   .when(col("Age") < 60, "40-59")
   .otherwise("60+")
).groupBy("age_bucket") \
 .agg({"new_credit_history_months": "avg", "new_loan_to_income_ratio": "avg"}) \
 .show()


# ## EDA : Feature Combination Heatmap (Correlation)

# In[25]:


numerics = [
    "Age", "Annual_Income", "Outstanding_Debt", "Monthly_Balance",
    "new_loan_to_income_ratio", "new_credit_history_months",
    "new_salary_debt_ratio", "new_inquiry_to_loan_ratio"
]
df_pd = df.select(numerics).toPandas()

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))
sns.heatmap(df_pd.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()


# ## Build gold table for labels

# In[26]:


# create bronze datalake
gold_feature_store_directory = "datamart/gold/features_store/"

if not os.path.exists(gold_feature_store_directory):
    os.makedirs(gold_feature_store_directory)


# In[27]:


# run gold backfill
for date_str in dates_str_lst:
    utils.data_processing_gold_table.process_features_gold_table(date_str,silver_features_clickstream_directory, silver_features_directory, gold_feature_store_directory, spark)


# In[28]:


utils.data_processing_gold_table.process_features_gold_table(date_str,silver_features_clickstream_directory, silver_features_directory, gold_feature_store_directory, spark).dtypes


# ## inspect feature store

# In[29]:


folder_path = gold_feature_store_directory
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
df = spark.read.option("header", "true").parquet(*files_list)
print("row_count:",df.count())

df.show()


# In[30]:


df.printSchema()





