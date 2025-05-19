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
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to silver table
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df


from datetime import datetime

'''def process_features_gold_table(snapshot_date_str, silver_features_clickstream_directory, silver_features_directory, gold_label_store_directory, spark):
    # Prepare snapshot date
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    partition_name = "silver_features_" + snapshot_date_str.replace('-', '_') + '.parquet'
    partition_name_clickstream = "silver_features_clickstream_" + snapshot_date_str.replace('-', '_') + '.parquet'

    # Load main features
    features_path = silver_features_directory + partition_name
    df_features = spark.read.parquet(features_path)
    print('Loaded features from:', features_path, 'row count:', df_features.count())

    # Load clickstream features
    clickstream_path = silver_features_clickstream_directory + partition_name_clickstream
    df_clickstream = spark.read.parquet(clickstream_path)
    print('Loaded clickstream from:', clickstream_path, 'row count:', df_clickstream.count())

    # Join on Customer_ID and snapshot_date
    df_joined = df_features.join(
        df_clickstream,
        on=["Customer_ID", "snapshot_date"],
        how="inner"
    )

    print("Joined row count:", df_joined.count())

    # Select columns from both sources
    selected_cols = [
        "Customer_ID",
        "snapshot_date",
        "Age",
        "Annual_Income",
        "Monthly_Inhand_Salary",
        "Num_Bank_Accounts",
        "Num_Credit_Card",
        "Interest_Rate",
        "Num_of_Loan",
        "Delay_from_due_date",
        "Num_of_Delayed_Payment",
        "Changed_Credit_Limit",
        "Num_Credit_Inquiries",
        "Outstanding_Debt",
        "Credit_Utilization_Ratio",
        "Total_EMI_per_month",
        "Amount_invested_monthly",
        "Monthly_Balance",
        "new_loan_to_income_ratio",
        "new_loan_type_count",
        "new_salary_debt_ratio",
        "new_inquiry_to_loan_ratio",
        "new_credit_history_months",
        "new_Payment_Behaviour_code",
        "new_Payment_of_Min_Amount_code",
        "new_Credit_Mix_code",
        "new_Occupation_code"
    ] + [f"fe_{i}" for i in range(1, 21)]  # Assuming fe_1 to fe_20 are from clickstream

    df_gold = df_joined.select(*selected_cols)

    # Save to gold path
    gold_partition_name = "gold_features_store_" + snapshot_date_str.replace('-', '_') + '.parquet'
    gold_filepath = gold_label_store_directory + gold_partition_name
    df_gold.write.mode("overwrite").parquet(gold_filepath)
    print('Saved gold features to:', gold_filepath)

    return df_gold'''

from datetime import datetime

def process_features_gold_table(
    snapshot_date_str,
    silver_features_clickstream_directory,
    silver_features_directory,
    gold_label_store_directory,
    spark
):
    # Prepare snapshot date & paths
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    p = snapshot_date_str.replace('-', '_')
    features_path     = f"{silver_features_directory}silver_features_{p}.parquet"
    clickstream_path  = f"{silver_features_clickstream_directory}silver_features_clickstream_{p}.parquet"

    # Load silver tables
    df_features    = spark.read.parquet(features_path)
    df_clickstream = spark.read.parquet(clickstream_path)

    print("Features rows:   ", df_features.count())
    print("Clickstream rows:", df_clickstream.count())

    # LEFT join → keep ALL features; missing clickstream → nulls
    df_joined = df_features.alias("f") \
        .join(
            df_clickstream.alias("c"),
            on=["Customer_ID", "snapshot_date"],
            how="left"
        )

    print("After join rows:", df_joined.count())  # should equal df_features.count()

    # pick your columns + fe_1…fe_20 from the clickstream side
    selected_cols = [
        "Customer_ID", "snapshot_date",
        "Age", "Annual_Income", "Monthly_Inhand_Salary",
        "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate",
        "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
        "Changed_Credit_Limit", "Num_Credit_Inquiries", "Outstanding_Debt",
        "Credit_Utilization_Ratio", "Total_EMI_per_month",
        "Amount_invested_monthly", "Monthly_Balance",
        "new_loan_to_income_ratio", "new_loan_type_count",
        "new_salary_debt_ratio", "new_inquiry_to_loan_ratio",
        "new_credit_history_months",
        "new_Payment_Behaviour_code", "new_Payment_of_Min_Amount_code",
        "new_Credit_Mix_code", "new_Occupation_code"
    ] + [f"fe_{i}" for i in range(1, 21)]

    df_gold = df_joined.select(*selected_cols)

    # Write out gold table
    gold_path = f"{gold_label_store_directory}gold_features_store_{p}.parquet"
    df_gold.write.mode("overwrite").parquet(gold_path)
    print("Saved gold features to:", gold_path)

    return df_gold
