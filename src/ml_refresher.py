# Databricks notebook source
import mlflow

mlflow.autolog(disable=True)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from bronze.fitness_tracker_data

# COMMAND ----------

# DBTITLE 1,Numerical Simple Monitoring Statistics
from pyspark.sql.functions import expr, max, mean, min, stddev, variance
 
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
    max("Calories_Burned").alias("Calories_Burned_Max"),
)

display(stats)

# COMMAND ----------

# DBTITLE 1,Categorical Simple Monitoring Statistics
from pyspark.sql.functions import col, countDistinct, expr

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
import matplotlib.pyplot as plt
import mlflow
import shap
from mlflow.models import infer_signature
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer, VectorAssembler

# Load data
df = spark.table("bronze.fitness_tracker_data")


# Using mlflow to track the model
with mlflow.start_run():
    # Prepare features and label
    indexer = StringIndexer(inputCol="Workout_Type", outputCol="label")
    df_indexed = indexer.fit(df).transform(df)

    # Assemble features
    assembler = VectorAssembler(inputCols=["Calories_Burned"], outputCol="features")
    df_features = assembler.transform(df_indexed)

    # Split data into training and test sets
    train_df, test_df = df_features.randomSplit([0.8, 0.2], seed=42)

    # Initialize Logistic Regression model
    lr = LogisticRegression(featuresCol="features", labelCol="label")
    # Train the model
    model = lr.fit(train_df)

    # Log a parameter (example: number of trees, though not applicable for Logistic Regression)
    mlflow.log_param("num_trees", 10)

    # Make predictions on the test set
    predictions = model.transform(test_df)

    # Calculate accuracy
    accuracy = predictions.filter(
        predictions.label == predictions.prediction,
    ).count() / float(test_df.count())

    # Log the accuracy metric
    mlflow.log_metric("accuracy", accuracy)

    # Infer the model signature
    signature = infer_signature(train_df, predictions)

    # Log the model
    mlflow.spark.log_model(model, "learning_model", signature=signature)

# COMMAND ----------

# DBTITLE 1,Predicts Calories Burned
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql.functions import col

# Load data
df = spark.table("bronze.fitness_tracker_data").select("Steps", "Heart_Rate_avg", "Workout_Type", "Calories_Burned")

# Handle missing values
df = df.dropna()

# Filter out invalid data
df = df.filter((col("Steps") >= 0) & (col("Heart_Rate_avg") > 0) & (col("Calories_Burned") >= 0))

# Index categorical column
indexer = StringIndexer(inputCol="Workout_Type", outputCol="Workout_Type_Indexed", handleInvalid="skip")
encoder = OneHotEncoder(inputCol="Workout_Type_Indexed", outputCol="Workout_Type_OHE", handleInvalid="keep")

# Assemble features
assembler = VectorAssembler(inputCols=["Steps", "Heart_Rate_avg", "Workout_Type_OHE"], outputCol="features", handleInvalid="skip")

# Define regressor
rf = RandomForestRegressor(featuresCol="features", labelCol="Calories_Burned", seed=42)

# Create pipeline
pipeline = Pipeline(stages=[indexer, encoder, assembler, rf])

# Split data into training and test sets
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Train model
model = pipeline.fit(train_df)

# Make predictions on the entire dataset
predictions = model.transform(df)

# Display predictions vs actual Calories_Burned
display(predictions.withColumn("Diff", col("Calories_Burned") - col("prediction")))

# COMMAND ----------

# DBTITLE 1,Linear Regression Model Evaluation Metrics
from pyspark.ml.evaluation import RegressionEvaluator

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

# Split data into training and test sets
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Train model
model = pipeline.fit(train_df)

# Make predictions
predictions = model.transform(test_df)

# Initialize evaluators
evaluator_rmse = RegressionEvaluator(predictionCol="prediction", labelCol="Calories_Burned", metricName="rmse")
evaluator_r2 = RegressionEvaluator(predictionCol="prediction", labelCol="Calories_Burned", metricName="r2")
evaluator_mae = RegressionEvaluator(predictionCol="prediction", labelCol="Calories_Burned", metricName="mae")

# Calculate metrics
rmse = evaluator_rmse.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)
mae = evaluator_mae.evaluate(predictions)

# Display metrics
metrics_df = spark.createDataFrame([(rmse, r2, mae)], ["RMSE", "R2", "MAE"])
display(metrics_df)

# COMMAND ----------

# DBTITLE 1,Predicts Workout Type
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType

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

# label map
label_map = {i: label for i, label in enumerate(model.stages[0].labels)}

# UDF to map prediction to label
map_prediction_udf = udf(lambda x: label_map.get(int(x), "Unknown"), StringType())

# Transform and add predicted label column
predictions = model.transform(df).withColumn("predicted_label", map_prediction_udf(col("prediction")))

display(predictions.select("Steps", "Heart_Rate_avg", "Calories_Burned", "Workout_Type", "prediction", "predicted_label"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Shap Test

# COMMAND ----------

# DBTITLE 1,SHAP Summary Plot for Model Predictions
import numpy as np
from pyspark.ml.linalg import DenseVector


# Define prediction function for SHAP explainer
# Converts numpy array to Spark DataFrame, applies model, and returns class probabilities
def predict_fn(data_asarray):
    features = [DenseVector(row) for row in data_asarray]
    spark_df = spark.createDataFrame([(f,) for f in features], ["features"])
    preds = rf_model.transform(spark_df).select("probability").toPandas()
    # Return probability of positive class (index 0); adjust index for other classes
    return np.array([p[0] for p in preds["probability"]])

# Prepare background and test datasets for SHAP
background_X = X[:100]  # Background dataset for Kernel SHAP
test_X = X[:50]         # Test dataset to explain

# Initialize SHAP KernelExplainer with prediction function and background data
explainer = shap.KernelExplainer(predict_fn, background_X)

# Compute SHAP values for test data; returns list of arrays (one per class) for classification
shap_values = explainer.shap_values(test_X)

# Plot SHAP summary plot
# For binary classification, plot SHAP values of positive class (index 1)
if isinstance(shap_values, list) and len(shap_values) == 2:
    shap.summary_plot(shap_values[1], test_X, feature_names=feature_cols)
else:
    shap.summary_plot(shap_values, test_X, feature_names=feature_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Kolmogorov-Smirnov (KS) Test

# COMMAND ----------

# DBTITLE 1,Normality Test on Fitness Tracker Data Columns
from scipy.stats import kstest

# Load data
df = spark.table("bronze.fitness_tracker_data").select("Steps", "Heart_Rate_avg", "Calories_Burned", "Workout_Type")
columns_to_test = ["Steps", "Heart_Rate_avg", "Calories_Burned"]

# Perform KS test for normality on specified columns
for col_name in columns_to_test:
    data = df.select(col_name).toPandas()[col_name]
    mean = np.mean(data)
    std = np.std(data)
    ks_stat, p_value = kstest(data, "norm", args=(mean, std))

    # Print KS test results
    print(f"{col_name} - KS Statistic: {ks_stat:.3f}, p-value: {p_value:.3g}")
    if p_value < 0.05:
        print(f"{col_name} is NOT normally distributed")
    else:
        print(f"{col_name} looks normal-ish")

# COMMAND ----------

# DBTITLE 1,Kolmogorov-Smirnov Test for Workout Type Distributions
import pandas as pd
from scipy.stats import ks_2samp

df = spark.table("bronze.fitness_tracker_data").select("Steps", "Heart_Rate_avg", "Calories_Burned", "Workout_Type")
workout_types = ["Cardio", "Strength", "Yoga", "None"]
columns_to_test = ["Steps", "Heart_Rate_avg", "Calories_Burned"]

results = []

for col_name in columns_to_test:
    for i in range(len(workout_types)):
        for j in range(i + 1, len(workout_types)):
            d1 = df.filter(df.Workout_Type == workout_types[i]).toPandas()[col_name]
            d2 = df.filter(df.Workout_Type == workout_types[j]).toPandas()[col_name]
            ks_stat, p_value = ks_2samp(d1, d2)
            results.append({
                "Comparison": f"{workout_types[i]} vs {workout_types[j]}",
                "Column": col_name,
                "KS_statistic": ks_stat,
                "p_value": p_value,
            })

# Highly significant (p < 0.05) AND KS_stat > 0.1-0.2: Distributions are different in a way that is likely to be meaningful.
# p >= 0.05: You can't claim there's a significant difference.

results_df = pd.DataFrame(results)
display(results_df)
# For your data, none of the fitness variables (Steps, Heart_Rate_avg, Calories_Burned) are statistically different in their distribution between any pair of workout types, at the 0.05 significance level.

# COMMAND ----------

# DBTITLE 1,Jensen-Shannon Divergence Between Workout Types
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

df = spark.table("bronze.fitness_tracker_data").select("Steps", "Heart_Rate_avg", "Calories_Burned", "Workout_Type")
workout_types = ["Cardio", "Strength", "Yoga", "None"]
columns_to_test = ["Steps", "Heart_Rate_avg", "Calories_Burned"]

results_jsd = []

for col in columns_to_test:
    for i in range(len(workout_types)):
        for j in range(i + 1, len(workout_types)):
            # Extract values for each group
            data_1 = df.filter(df.Workout_Type == workout_types[i]).toPandas()[col]
            data_2 = df.filter(df.Workout_Type == workout_types[j]).toPandas()[col]

            # Remove NaNs if present
            data_1 = data_1.dropna()
            data_2 = data_2.dropna()

            # Bin the data to create comparable distributions (histograms)
            min_val = min(data_1.min(), data_2.min())
            max_val = max(data_1.max(), data_2.max())
            bins = np.linspace(min_val, max_val, 31)  # 30 bins

            hist_1, _ = np.histogram(data_1, bins=bins, density=True)
            hist_2, _ = np.histogram(data_2, bins=bins, density=True)

            # Add small epsilon for 0–avoid log(0)
            epsilon = 1e-12
            hist_1 += epsilon
            hist_2 += epsilon

            # Normalize histograms to probability distributions
            p = hist_1 / np.sum(hist_1)
            q = hist_2 / np.sum(hist_2)

            # Compute JSD (returns square root of divergence for scipy >=1.6.0)
            jsd = jensenshannon(p, q, base=2)
            divergence = jsd**2  # True JSD, in [0,1]; jsd is sqrt(JSD)

            results_jsd.append({
                "Comparison": f"{workout_types[i]} vs {workout_types[j]}",
                "Column": col,
                "JSD": divergence,  # use jsd if you want sqrt(JSD), divergence for JSD proper
            })

# Results to DataFrame
results_df = pd.DataFrame(results_jsd)
def interpret_jsd(jsd):
    if jsd < 0.05:
        return "Very similar"
    elif jsd < 0.2:
        return "Somewhat different"
    else:
        return "Noticeably different"

results_df["Divergence"] = results_df["JSD"].apply(interpret_jsd)
display(results_df)

# COMMAND ----------

# DBTITLE 1,KDE Plots of Fitness Data by Workout Type
import seaborn as sns

df = spark.table("bronze.fitness_tracker_data").select("Steps", "Heart_Rate_avg", "Calories_Burned", "Workout_Type")
workout_types = ["Cardio", "Strength", "Yoga", "None"]
columns_to_test = ["Steps", "Heart_Rate_avg", "Calories_Burned"]

for col in columns_to_test:
    for w in workout_types:
        sample = df.filter(df.Workout_Type == w).toPandas()[col]
        sns.kdeplot(sample, label=w)

    plt.title(f"Distribution of {col} by Workout Type")
    plt.xlabel(col)
    plt.legend()
    plt.show()

# COMMAND ----------

#  Chi-square Test
