# Databricks notebook source
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression

# Load data
df = spark.table("silver.fitness_tracker_data")

# Prepare features and label
indexer = StringIndexer(inputCol="Workout_Type", outputCol="label")
df_indexed = indexer.fit(df).transform(df)

assembler = VectorAssembler(inputCols=["avg_calories"], outputCol="features")
df_features = assembler.transform(df_indexed)

# Split data
train_df, test_df = df_features.randomSplit([0.8, 0.2], seed=42)

# Train model
lr = LogisticRegression(featuresCol="features", labelCol="label")
model = lr.fit(train_df)

# Display model summary
display(model.summary)

# COMMAND ----------

# TODO
#  Kolmogorov-Smirnov (KS) Test
#  Jensen-Shannon Divergence (JSD)
#  Chi-square Test
