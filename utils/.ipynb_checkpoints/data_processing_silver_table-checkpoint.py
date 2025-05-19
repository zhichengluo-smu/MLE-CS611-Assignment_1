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

from pyspark.sql.functions import col , regexp_replace, when, trim , regexp_extract, col , lit,size, split
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import matplotlib.pyplot as plt
import pandas as pd

def add_encode_occupation(df,
                          src_col: str = "Occupation",
                          dst_col: str = "new_Occupation_code"):
    return df.withColumn(
        dst_col,
        when(col(src_col) == "Accountant",      lit(0))
       .when(col(src_col) == "Architect",       lit(1))
       .when(col(src_col) == "Developer",       lit(2))
       .when(col(src_col) == "Doctor",          lit(3))
       .when(col(src_col) == "Engineer",        lit(4))
       .when(col(src_col) == "Entrepreneur",    lit(5))
       .when(col(src_col) == "Journalist",      lit(6))
       .when(col(src_col) == "Lawyer",          lit(7))
       .when(col(src_col) == "Manager",         lit(8))
       .when(col(src_col) == "Mechanic",        lit(9))
       .when(col(src_col) == "Media_Manager",   lit(10))
       .when(col(src_col) == "Musician",        lit(11))
       .when(col(src_col) == "Scientist",       lit(12))
       .when(col(src_col) == "Teacher",         lit(13))
       .when(col(src_col) == "Writer",          lit(14))
       .otherwise(lit(None))
       .cast(IntegerType())
    )

def add_encode_credit_score(df,
                            src_col: str = "Credit_Mix",
                            dst_col: str = "new_Credit_Mix_code"):
    return df.withColumn(
        dst_col,
        when(col(src_col) == "Good",     lit(3))
       .when(col(src_col) == "Standard", lit(2))
       .when(col(src_col) == "Bad",      lit(1))
       .otherwise(lit(None))
       .cast(IntegerType())
    )

def add_encode_min_payment(df,
                           src_col: str = "Payment_of_Min_Amount",
                           dst_col: str = "new_Payment_of_Min_Amount_code"):
    return df.withColumn(
        dst_col,
        when(col(src_col) == "NM",  lit(0))
       .when(col(src_col) == "Yes", lit(1))
       .when(col(src_col) == "No",  lit(2))
       .otherwise(lit(None))
       .cast(IntegerType())
    )

def add_encode_payment_behaviour(df,
                                 src_col: str = "Payment_Behaviour",
                                 dst_col: str = "new_Payment_Behaviour_code"):
    return df.withColumn(
        dst_col,
        when(col(src_col) == "Low_spent_Small_value_payments",   lit(1))
       .when(col(src_col) == "Low_spent_Medium_value_payments", lit(2))
       .when(col(src_col) == "Low_spent_Large_value_payments",  lit(3))
       .when(col(src_col) == "High_spent_Small_value_payments", lit(4))
       .when(col(src_col) == "High_spent_Medium_value_payments",lit(5))
       .when(col(src_col) == "High_spent_Large_value_payments", lit(6))
       .otherwise(lit(None))
       .cast(IntegerType())
    )

def add_credit_history_months(df,
                              src_col: str = "Credit_History_Age",
                              new_col: str = "new_credit_history_months"):
    # extract the “Years” and “Months” parts as ints
    years = regexp_extract(col(src_col), r"(\d+)\s+Years", 1).cast(IntegerType())
    months = regexp_extract(col(src_col), r"and\s+(\d+)\s+Months", 1).cast(IntegerType())
    # compute total months and attach as a new column
    return df.withColumn(new_col, years * 12 + months)

def nullifies_outliers_via_boxplot(df, column, spark, return_spark_df=True):
    """
    1. Converts the specified Spark DataFrame column into a Pandas Series to calculate IQR.
    2. Computes IQR-based bounds.
    3. Nullifies outlier values in the specified column (keeps the row).
    4. Returns the updated Spark or Pandas DataFrame.
    """
    # Convert Spark column to Pandas to calculate IQR
    pdf = df.select(column).dropna().toPandas()
    
    Q1 = pdf[column].quantile(0.25)
    Q3 = pdf[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Nullify outliers in Spark DataFrame
    df = df.withColumn(
        column,
        when((col(column) < lower_bound) | (col(column) > upper_bound), lit(None)) \
        .otherwise(col(column).cast(FloatType()))
    )

    if return_spark_df:
        return df
    else:
        return df.select(column).toPandas()

def clean_Attribute(df):
    """
    Standard clean-up for both attribute and financial columns in the silver layer.
    Applies column-specific rules:
      - Name, SSN, Occupation, Age
      - Annual_Income, Num_Bank_Accounts, Num_Credit_Card, Interest_Rate,
        Num_of_Loan, Num_of_Delayed_Payment, Changed_Credit_Limit,
        Credit_Mix, Outstanding_Debt, Amount_invested_monthly, Payment_Behaviour
    """
    # ---- Attribute columns ----
    # Name: letters, spaces, basic punctuation
    df = df.withColumn(
        "Name",
        trim(regexp_replace(col("Name"), r"[^A-Za-z\s\.'-]", ""))
    )
    # Age: digits only, valid 0-120
    df = df.withColumn(
        "Age_tmp",
        regexp_replace(col("Age").cast(StringType()), r"[^0-9]", "").cast(IntegerType())
    ).withColumn(
        "Age",
        when((col("Age_tmp") >= 0) & (col("Age_tmp") <= 120), col("Age_tmp")).otherwise(None)
    ).drop("Age_tmp")
    # SSN: digits and dashes, enforce XXX-XX-XXXX
    df = df.withColumn(
        "SSN_tmp",
        regexp_replace(col("SSN"), r"[^0-9-]", "")
    ).withColumn(
        "SSN",
        when(col("SSN_tmp").rlike(r"^\d{3}-\d{2}-\d{4}$"), col("SSN_tmp")).otherwise(None)
    ).drop("SSN_tmp")
    # Occupation: null out blanks or underscores only
    df = df.withColumn(
        "Occupation",
        when(trim(col("Occupation")) == "", None)
        .when(col("Occupation").rlike(r"^_+$"), None)
        .otherwise(col("Occupation"))
    )
    
    return df
    
def clean_Financial(df):
    # ---- Financial columns ----
    # Annual Income: strip underscores
    df = df.withColumn(
        "Annual_Income",
        regexp_replace(col("Annual_Income").cast(StringType()), r"_+", "").cast(FloatType())
    )
    # Num_Bank_Accounts: int <100
    df = df.withColumn(
        "Num_Bank_Accounts",
        when(col("Num_Bank_Accounts").cast(IntegerType()) < 100,
             col("Num_Bank_Accounts").cast(IntegerType()))
        .otherwise(None)
    )
    # Num_Credit_Card: int <50
    df = df.withColumn(
        "Num_Credit_Card",
        when(col("Num_Credit_Card").cast(IntegerType()) < 50,
             col("Num_Credit_Card").cast(IntegerType()))
        .otherwise(None)
    )
    # Interest_Rate: percent 0-99
    df = df.withColumn(
        "Interest_Rate",
        when((col("Interest_Rate").cast(IntegerType()) >= 0) &
             (col("Interest_Rate").cast(IntegerType()) < 100),
             col("Interest_Rate").cast(IntegerType()))
        .otherwise(None)
    )
    # Clean Num_of_Loan: remove underscores first, then validate 0-19
    df = df.withColumn(
        "Num_of_Loan_tmp",
        regexp_replace(col("Num_of_Loan").cast(StringType()), r"_+", "")
    ).withColumn(
        "Num_of_Loan",
        when(
            (col("Num_of_Loan_tmp").cast(IntegerType()) >= 0) & 
            (col("Num_of_Loan_tmp").cast(IntegerType()) < 20),
            col("Num_of_Loan_tmp").cast(IntegerType())
        ).otherwise(None)
    ).drop("Num_of_Loan_tmp")

    # Clean Num_of_Delayed_Payment: remove underscores first, then validate 0-29
    df = df.withColumn(
        "Num_of_Delayed_Payment_tmp",
        regexp_replace(col("Num_of_Delayed_Payment").cast(StringType()), r"_+", "")
    ).withColumn(
        "Num_of_Delayed_Payment",
        when(
            (col("Num_of_Delayed_Payment_tmp").cast(IntegerType()) >= 0) & 
            (col("Num_of_Delayed_Payment_tmp").cast(IntegerType()) < 30),
            col("Num_of_Delayed_Payment_tmp").cast(IntegerType())
        ).otherwise(None)
    ).drop("Num_of_Delayed_Payment_tmp")

    # Changed_Credit_Limit: strip underscores, null if empty
    df = df.withColumn(
        "Changed_Credit_Limit_tmp",
        regexp_replace(col("Changed_Credit_Limit"), r"_+", "")
    ).withColumn(
        "Changed_Credit_Limit",
        when(trim(col("Changed_Credit_Limit_tmp")) == "", None)
        .otherwise(col("Changed_Credit_Limit_tmp").cast(FloatType()))
    ).drop("Changed_Credit_Limit_tmp")
    # Credit_Mix: null if blank/underscores
    df = df.withColumn(
        "Credit_Mix",
        when(trim(col("Credit_Mix")) == "", None)
        .when(col("Credit_Mix").rlike(r"^_+$"), None)
        .otherwise(col("Credit_Mix"))
    )
    # Outstanding_Debt: strip underscores
    df = df.withColumn(
        "Outstanding_Debt",
        regexp_replace(col("Outstanding_Debt").cast(StringType()), r"_+", "").cast(FloatType())
    )
    # Amount_invested_monthly: digits and dot only
    df = df.withColumn(
        "Amount_invested_monthly",
        regexp_replace(col("Amount_invested_monthly"), r"[^0-9.]", "").cast(FloatType())
    )
    
    # Payment_Behaviour: Only the exact match of "!@9#%8" will be replaced with None.
    df = df.withColumn(
        "Payment_Behaviour",
        when(col("Payment_Behaviour") == "!@9#%8", None).otherwise(col("Payment_Behaviour"))
    )


    return df

    

def process_silver_table(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # save silver table - IRL connect to database to write
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_silver_features_table(snapshot_date_str,
                            bronze_attr_dir,
                            bronze_fin_dir,
                            silver_feature_dir,
                            spark):
    # parse date
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    part_suffix = snapshot_date_str.replace('-', '_')
    
    # 1. load both bronze tables
    attr_path = os.path.join(bronze_attr_dir, f"bronze_features_attributes_{part_suffix}.csv")
    fin_path  = os.path.join(bronze_fin_dir,  f"bronze_features_financials_{part_suffix}.csv")
    
    df_attr = spark.read.csv(attr_path, header=True, inferSchema=True)
    df_fin  = spark.read.csv(fin_path,  header=True, inferSchema=True)
    
    df_attr = clean_Attribute(df_attr) 
    df_fin  = clean_Financial(df_fin)

    df_fin = nullifies_outliers_via_boxplot(df_fin, "Annual_Income", spark)
    df_fin = nullifies_outliers_via_boxplot(df_fin, "Num_Credit_Inquiries", spark)
    # 2. enforce schema on common & unique cols
    #    adjust types to match your downstream expectations
    attr_types = {
        "Customer_ID": StringType(),
        "Name":        StringType(),
        "Age":         IntegerType(),
        "SSN":         StringType(),
        "Occupation":  StringType(),
        "snapshot_date": DateType()
    }
    fin_types = {
        "Customer_ID":             StringType(),
        "Annual_Income":           FloatType(),
        "Monthly_Inhand_Salary":   FloatType(),
        "Num_Bank_Accounts":       IntegerType(),
        "Num_Credit_Card":         IntegerType(),
        "Interest_Rate":           IntegerType(),
        "Num_of_Loan":             IntegerType(),
        "Type_of_Loan":            StringType(),
        "Delay_from_due_date":     IntegerType(),
        "Num_of_Delayed_Payment":  IntegerType(),
        "Changed_Credit_Limit":    FloatType(),
        "Num_Credit_Inquiries":    IntegerType(),
        "Credit_Mix":              StringType(),
        "Outstanding_Debt":        FloatType(),
        "Credit_Utilization_Ratio": FloatType(),
        "Credit_History_Age":      StringType(),
        "Payment_of_Min_Amount":   StringType(),
        "Total_EMI_per_month":     FloatType(),
        "Amount_invested_monthly": FloatType(),
        "Payment_Behaviour":       StringType(),
        "Monthly_Balance":         FloatType(),
        "snapshot_date":           DateType()
    }
    
    for col_name, dtype in attr_types.items():
        df_attr = df_attr.withColumn(col_name, col(col_name).cast(dtype))
    for col_name, dtype in fin_types.items():
        df_fin = df_fin.withColumn(col_name, col(col_name).cast(dtype))
    
    # join on customer + snapshot_date
    df = (df_attr
          .join(df_fin, on=["Customer_ID", "snapshot_date"], how="inner"))
    
    # loan-to-income 
    df = df.withColumn(
        "new_loan_to_income_ratio",
        (col("Outstanding_Debt") / col("Annual_Income")).cast(FloatType())
    )
    # loan_type_count: how many distinct loan types per customer
    df = df.withColumn(
        "new_loan_type_count",
        size(split(col("Type_of_Loan"), r"\s*,\s*")).cast(IntegerType())
    )
    # salary_debt_ratio: how much they owe vs take-home pay
    df = df.withColumn(
        "new_salary_debt_ratio",
        (col("Outstanding_Debt") / (col("Monthly_Inhand_Salary") + lit(1e-6)))
          .cast(FloatType())
    )
    # Inquiry-to-loan ratio: frequent credit pulls vs. number of loans
    df = df.withColumn(
        "new_inquiry_to_loan_ratio",
        (col("Num_Credit_Inquiries") / (col("Num_of_Loan") + lit(1e-6)))
          .cast(FloatType())
    )
    
    
    # credit_history_months transform
    df = add_credit_history_months(df)

    # payment_behaviour transform
    df = add_encode_payment_behaviour(df)

    # Payment_of_Min_Amount transform
    df = add_encode_min_payment(df)

    # add_encode_credit_score transform
    df = add_encode_credit_score(df)

    # add_encode_occupation transform
    df = add_encode_occupation(df)

    

    # 5. write out silver feature table
    out_path = os.path.join(silver_feature_dir, f"silver_features_{part_suffix}.parquet")
    df.write.mode("overwrite").parquet(out_path)
    print(f"Silver features written to: {out_path}  (rows: {df.count()})")
    
    return df

def process_silver_features_clickstream_table(snapshot_date_str, bronze_clickstream_directory, silver_features_clickstream_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_features_clickstream_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_clickstream_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())


    for col_name in [f"fe_{i}" for i in range(1, 21)]:
        df = df.withColumn(col_name, df[col_name].cast(IntegerType()))

    # save silver table - IRL connect to database to write
    partition_name = "silver_features_clickstream_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_features_clickstream_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    
    return df
