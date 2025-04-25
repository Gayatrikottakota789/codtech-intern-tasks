# PySpark Code for Big Data Analysis

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, count, desc

# Step 1: Initialize Spark session
spark = SparkSession.builder \
    .appName("Big Data Sales Analysis") \
    .getOrCreate()

# Step 2: Load the CSV file
df = spark.read.csv("sales_data(task-1).csv", header=True, inferSchema=True)

# Step 3: Basic Data Exploration
print("\nSchema:")
df.printSchema()
print("\nSample Data:")
df.show(5)

# Step 4: Total Sales Amount
total_sales = df.agg(sum("Amount").alias("Total_Sales"))
total_sales.show()

# Step 5: Sales by Category
sales_by_category = df.groupBy("Category").agg(
    sum("Amount").alias("Total_Sales"),
    count("OrderID").alias("Number_of_Orders")
)
sales_by_category.orderBy(desc("Total_Sales")).show()

# Step 6: Top 5 Cities by Revenue
top_cities = df.groupBy("City").agg(sum("Amount").alias("Revenue"))

top_cities.orderBy(desc("Revenue")).show(5)


# Stop the Spark session
spark.stop()
