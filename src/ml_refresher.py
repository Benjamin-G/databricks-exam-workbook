# Databricks notebook source
# MAGIC %sql
# MAGIC select * from bronze.fitness_tracker_data

# COMMAND ----------

# DBTITLE 1,Numerical Simple Monitoring Statistics
from pyspark.sql.functions import mean, expr, stddev, variance, min, max, percentile_approx

df = spark.table("bronze.fitness_tracker_data").select("Steps", "Heart_Rate_avg", "Calories_Burned")

stats = df.agg(
    mean("Steps").alias("Steps_Mean"),
    expr("percentile_approx(Steps, 0.5)").alias("Steps_Median"),
    stddev("Steps").alias("Steps_StdDev"),
    variance("Steps").alias("Steps_Variance"),
    min("Steps").alias("Steps_Min"),
    max("Steps").alias("Steps_Max"),
    
    mean("Heart_Rate_avg").alias("Heart_Rate_avg_Mean"),
    expr("percentile_approx(Heart_Rate_avg, 0.5)").alias("Heart_Rate_avg_Median"),
    stddev("Heart_Rate_avg").alias("Heart_Rate_avg_StdDev"),
    variance("Heart_Rate_avg").alias("Heart_Rate_avg_Variance"),
    min("Heart_Rate_avg").alias("Heart_Rate_avg_Min"),
    max("Heart_Rate_avg").alias("Heart_Rate_avg_Max"),
    
    mean("Calories_Burned").alias("Calories_Burned_Mean"),
    expr("percentile_approx(Calories_Burned, 0.5)").alias("Calories_Burned_Median"),
    stddev("Calories_Burned").alias("Calories_Burned_StdDev"),
    variance("Calories_Burned").alias("Calories_Burned_Variance"),
    min("Calories_Burned").alias("Calories_Burned_Min"),
    max("Calories_Burned").alias("Calories_Burned_Max")
)

display(stats)

# COMMAND ----------

# DBTITLE 1,Categorical Simple Monitoring Statistics
from pyspark.sql.functions import col, count, countDistinct, expr

df = spark.table("bronze.fitness_tracker_data").select("Workout_Type")

# Calculate mode
mode = df.groupBy("Workout_Type").count().orderBy(col("count").desc()).first()["Workout_Type"]

# Count of unique values
unique_count = df.select(countDistinct("Workout_Type")).first()[0]

# Missing value rate
total_count = df.count()
missing_count = df.filter(col("Workout_Type").isNull()).count()
missing_rate = missing_count / total_count

# Create a DataFrame to display the results
result = spark.createDataFrame([(mode, unique_count, missing_rate)], ["Mode", "Unique_Count", "Missing_Rate"])

display(result)

# COMMAND ----------

# DBTITLE 1,Simple model
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression

# Load data
df = spark.table("bronze.fitness_tracker_data")

# Prepare features and label
indexer = StringIndexer(inputCol="Workout_Type", outputCol="label")
df_indexed = indexer.fit(df).transform(df)

assembler = VectorAssembler(inputCols=["Calories_Burned"], outputCol="features")
df_features = assembler.transform(df_indexed)

# Split data
train_df, test_df = df_features.randomSplit([0.8, 0.2], seed=42)

# Train model
lr = LogisticRegression(featuresCol="features", labelCol="label")
model = lr.fit(train_df)

# Display model summary
display(model.summary.predictions)

# COMMAND ----------

# DBTITLE 1,Predicts Calories Burned
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.sql.functions import col

# Load data
df = spark.table("bronze.fitness_tracker_data").select("Steps", "Heart_Rate_avg", "Workout_Type", "Calories_Burned")

# Handle missing values
df = df.dropna()

# Filter out invalid data
df = df.filter((col("Steps") >= 0) & (col("Heart_Rate_avg") > 0) & (col("Calories_Burned") >= 0))

# Index categorical column
indexer = StringIndexer(inputCol="Workout_Type", outputCol="Workout_Type_Indexed", handleInvalid="skip")

# Assemble features
assembler = VectorAssembler(inputCols=["Steps", "Heart_Rate_avg", "Workout_Type_Indexed"], outputCol="features", handleInvalid="skip")

# Define regressor
lr = LinearRegression(featuresCol="features", labelCol="Calories_Burned")

# Create pipeline
pipeline = Pipeline(stages=[indexer, assembler, lr])

# Train model
model = pipeline.fit(df)

# Display model summary
display(model.summary)

# COMMAND ----------

# DBTITLE 1,Predicts Workout Type
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.sql.functions import col

# Load data
df = spark.table("bronze.fitness_tracker_data").select("Steps", "Heart_Rate_avg", "Calories_Burned", "Workout_Type")

# Handle missing values
df = df.dropna()

# Filter out invalid data
df = df.filter((col("Steps") >= 0) & (col("Heart_Rate_avg") > 0) & (col("Calories_Burned") >= 0))

# Index label column
label_indexer = StringIndexer(inputCol="Workout_Type", outputCol="label", handleInvalid="skip")

# Assemble features
assembler = VectorAssembler(inputCols=["Steps", "Heart_Rate_avg", "Calories_Burned"], outputCol="features", handleInvalid="skip")

# Define classifier
rf = RandomForestClassifier(featuresCol="features", labelCol="label", seed=42)

# Create pipeline
pipeline = Pipeline(stages=[label_indexer, assembler, rf])

# Train model
model = pipeline.fit(df)

display(model.transform(df).select("Steps", "Heart_Rate_avg", "Calories_Burned", "Workout_Type", "prediction"))

# COMMAND ----------

# TODO
#  Kolmogorov-Smirnov (KS) Test
#  Jensen-Shannon Divergence (JSD)
#  Chi-square Test
