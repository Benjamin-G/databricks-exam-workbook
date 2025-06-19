# Databricks notebook source
# MAGIC %sql
# MAGIC describe extended bronze.fitness_tracker_data

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE silver.fitness_tracker_data SHALLOW CLONE bronze.fitness_tracker_data

# COMMAND ----------

# MAGIC %sql
# MAGIC select round(avg(Calories_Burned),2) as avg_calories, Workout_Type from silver.fitness_tracker_data group by Workout_Type

# COMMAND ----------

# MAGIC %sql
# MAGIC vacuum bronze.fitness_tracker_data 

# COMMAND ----------

# MERGE INTO
