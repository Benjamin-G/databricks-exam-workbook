# Databricks notebook source
import logging
from typing import Dict

import mlflow
import mlflow.spark
import pandas as pd
import seaborn as sns
from mlflow.models import infer_signature
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import StandardScaler, StringIndexer, VectorAssembler
from pyspark.sql.functions import col, count, udf, when
from pyspark.sql.types import StringType
 
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress py4j verbose logging
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)
logging.getLogger("py4j.clientserver").setLevel(logging.ERROR)


def validate_and_clean_data(df):
    """Validate and clean the input data"""
    logger.info("Validating and cleaning data...")

    # Initial data info
    initial_count = df.count()
    logger.info(f"Initial dataset size: {initial_count} rows")

    # Check for null values before cleaning
    null_counts = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).collect()[0]
    for col_name, null_count in null_counts.asDict().items():
        if null_count > 0:
            logger.info(f"Column {col_name} has {null_count} null values ({null_count / initial_count * 100:.2f}%)")

    # Handle missing values
    df_clean = df.dropna()
    after_dropna = df_clean.count()
    logger.info(
        f"After dropping nulls: {after_dropna} rows ({(initial_count - after_dropna) / initial_count * 100:.2f}% removed)")

    # Filter out invalid data
    df_clean = df_clean.filter(
        (col("Steps") >= 0) &
        (col("Heart_Rate_avg") > 0) &
        (col("Calories_Burned") >= 0),
    )

    final_count = df_clean.count()
    logger.info(
        f"After filtering invalid data: {final_count} rows ({(after_dropna - final_count) / after_dropna * 100:.2f}% removed)")

    # Show data quality summary
    logger.info("Data quality summary:")
    df_clean.describe().show()

    # Check class distribution
    logger.info("Workout type distribution:")
    class_dist = df_clean.groupBy("Workout_Type").count().orderBy("count", ascending=False)
    class_dist.show()

    return df_clean


def create_preprocessing_pipeline(feature_cols, target_col, scale_features=True):
    """Create preprocessing pipeline"""
    stages = []

    # Index label column
    label_indexer = StringIndexer(
        inputCol=target_col,
        outputCol="label",
        handleInvalid="skip",
    )
    stages.append(label_indexer)

    # Assemble features
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="raw_features" if scale_features else "features",
        handleInvalid="skip",
    )
    stages.append(assembler)

    # Optional feature scaling
    if scale_features:
        scaler = StandardScaler(
            inputCol="raw_features",
            outputCol="features",
            withStd=True,
            withMean=True,
        )
        stages.append(scaler)

    return stages
 

def evaluate_model(predictions, num_classes) -> Dict[str, float]:
    """Comprehensive model evaluation"""
    metrics = {}

    # Accuracy
    accuracy_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy",
    )
    metrics["accuracy"] = accuracy_evaluator.evaluate(predictions)

    # F1 Score
    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1",
    )
    metrics["f1_score"] = f1_evaluator.evaluate(predictions)

    # Precision
    precision_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="weightedPrecision",
    )
    metrics["precision"] = precision_evaluator.evaluate(predictions)

    # Recall
    recall_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="weightedRecall",
    )
    metrics["recall"] = recall_evaluator.evaluate(predictions)

    # AUC for binary classification
    if num_classes == 2:
        auc_evaluator = BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC",
        )
        metrics["auc_roc"] = auc_evaluator.evaluate(predictions)

    return metrics


def create_confusion_matrix_plot(predictions, label_map):
    """Create confusion matrix visualization"""
    try:
        # Convert to pandas for easier plotting
        pred_df = predictions.select("label", "prediction").toPandas()

        # Map numeric labels back to strings
        pred_df["actual_label"] = pred_df["label"].map(label_map)
        pred_df["predicted_label"] = pred_df["prediction"].map(label_map)

        # Create confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(pred_df["actual_label"], pred_df["predicted_label"])

        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=list(label_map.values()),
                    yticklabels=list(label_map.values()))
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
        plt.close()

        return "confusion_matrix.png"
    except Exception as e:
        logger.error(f"Error creating confusion matrix: {e!s}")
        return None


def create_feature_importance_plot(model, feature_cols):
    """Create feature importance visualization"""
    try:
        # Get feature importances from Random Forest
        rf_model = model.stages[-1]  # Last stage is the RF classifier
        importances = rf_model.featureImportances.toArray()

        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": importances,
        }).sort_values("importance", ascending=True)

        # Plot
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df["feature"], importance_df["importance"])
        plt.title("Feature Importance (Random Forest)")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=300, bbox_inches="tight")
        plt.close()

        return "feature_importance.png"
    except Exception as e:
        logger.error(f"Error creating feature importance plot: {e!s}")
        return None


def main():
    """Main training pipeline"""

    # Configuration
    FEATURE_COLS = ["Steps", "Heart_Rate_avg", "Calories_Burned"]
    TARGET_COL = "Workout_Type"
    TEST_SIZE = 0.2
    RANDOM_SEED = 42
    SCALE_FEATURES = True

    # Random Forest parameters
    RF_PARAMS = {
        "numTrees": 100,
        "maxDepth": 10,
        "minInstancesPerNode": 2,
        "subsamplingRate": 0.8,
        "featureSubsetStrategy": "auto",
    }

    try:
        # Load and validate data
        logger.info("Loading data...")
        df = spark.table("bronze.fitness_tracker_data").select(
            "Steps", "Heart_Rate_avg", "Calories_Burned", "Workout_Type",
        )

        df_clean = validate_and_clean_data(df)

        # Start MLflow run
        with mlflow.start_run() as run:
            logger.info(f"Started MLflow run: {run.info.run_id}")

            # Log parameters
            mlflow.log_param("feature_columns", FEATURE_COLS)
            mlflow.log_param("target_column", TARGET_COL)
            mlflow.log_param("test_size", TEST_SIZE)
            mlflow.log_param("random_seed", RANDOM_SEED)
            mlflow.log_param("scale_features", SCALE_FEATURES)

            for param, value in RF_PARAMS.items():
                mlflow.log_param(f"rf_{param}", value)

            # Split data
            logger.info("Splitting data...")
            train_df, test_df = df_clean.randomSplit([1 - TEST_SIZE, TEST_SIZE], seed=RANDOM_SEED)

            train_count = train_df.count()
            test_count = test_df.count()
            logger.info(f"Training samples: {train_count}, Test samples: {test_count}")

            mlflow.log_param("train_samples", train_count)
            mlflow.log_param("test_samples", test_count)

            # Create preprocessing pipeline
            preprocessing_stages = create_preprocessing_pipeline(
                FEATURE_COLS, TARGET_COL, SCALE_FEATURES,
            )

            # Define Random Forest classifier
            rf = RandomForestClassifier(
                featuresCol="features",
                labelCol="label",
                seed=RANDOM_SEED,
                **RF_PARAMS,
            )

            # Create complete pipeline
            pipeline = Pipeline(stages=preprocessing_stages + [rf])

            # Train model
            logger.info("Training model...")
            model = pipeline.fit(train_df)

            # Get label mapping
            label_indexer = model.stages[0]
            label_map = {i: label for i, label in enumerate(label_indexer.labels)}
            num_classes = len(label_map)

            logger.info(f"Number of classes: {num_classes}")
            logger.info(f"Class labels: {list(label_map.values())}")

            mlflow.log_param("num_classes", num_classes)
            mlflow.log_param("class_labels", list(label_map.values()))

            # Create UDF to map predictions back to labels
            map_prediction_udf = udf(lambda x: label_map.get(int(x), "Unknown"), StringType())

            # Make predictions
            logger.info("Making predictions...")
            train_predictions = model.transform(train_df).withColumn(
                "predicted_label", map_prediction_udf(col("prediction")),
            )
            test_predictions = model.transform(test_df).withColumn(
                "predicted_label", map_prediction_udf(col("prediction")),
            )

            # Evaluate model
            train_metrics = evaluate_model(train_predictions, num_classes)
            test_metrics = evaluate_model(test_predictions, num_classes)

            # Log metrics
            for metric_name, value in train_metrics.items():
                mlflow.log_metric(f"train_{metric_name}", value)
                logger.info(f"Train {metric_name}: {value:.4f}")

            for metric_name, value in test_metrics.items():
                mlflow.log_metric(f"test_{metric_name}", value)
                logger.info(f"Test {metric_name}: {value:.4f}")

            # Check for overfitting
            accuracy_diff = train_metrics["accuracy"] - test_metrics["accuracy"]
            mlflow.log_metric("overfitting_gap", accuracy_diff)

            if accuracy_diff > 0.15:
                logger.warning(f"Potential overfitting detected. Accuracy gap: {accuracy_diff:.4f}")

            # Create visualizations
            logger.info("Creating visualizations...")

            # Confusion matrix
            cm_file = create_confusion_matrix_plot(test_predictions, label_map)
            if cm_file:
                mlflow.log_artifact(cm_file)

            # Feature importance
            fi_file = create_feature_importance_plot(model, FEATURE_COLS)
            if fi_file:
                mlflow.log_artifact(fi_file)

            # Infer model signature
            signature = infer_signature(
                train_df.select(FEATURE_COLS).toPandas(),
                test_predictions.select("prediction", "predicted_label").toPandas(),
            )

            # Define pip requirements to speed up model logging
            pip_requirements = [
                "pyspark>=3.0.0",
                "mlflow>=2.0.0",
                "pandas>=1.3.0",
                "numpy>=1.20.0",
                "scikit-learn>=1.0.0",
            ]

            # Log model with explicit requirements
            mlflow.spark.log_model(
                model,
                "random_forest_model",
                signature=signature,
                pip_requirements=pip_requirements,
            )

            # Display sample predictions
            logger.info("Sample predictions:")
            sample_predictions = test_predictions.select(
                "Steps", "Heart_Rate_avg", "Calories_Burned",
                "Workout_Type", "predicted_label", "prediction",
            ).limit(20)

            sample_predictions.show()

            logger.info("Training pipeline completed successfully!")
            logger.info(f"Model test accuracy: {test_metrics['accuracy']:.4f}")

            return model, test_metrics, label_map

    except Exception as e:
        logger.error(f"Training pipeline failed: {e!s}")
        raise


model, metrics, label_map = main()

# Optional: Show final results
print("\n" + "=" * 50)
print("FINAL RESULTS")
print("=" * 50)
print(f"Test Accuracy: {metrics['accuracy']:.4f}")
print(f"Test F1-Score: {metrics['f1_score']:.4f}")
print(f"Test Precision: {metrics['precision']:.4f}")
print(f"Test Recall: {metrics['recall']:.4f}")
if "auc_roc" in metrics:
    print(f"Test AUC-ROC: {metrics['auc_roc']:.4f}")
print(f"Label Mapping: {label_map}")
print("=" * 50)

# COMMAND ----------

import logging
from typing import Dict, List

import matplotlib.pyplot as plt
import mlflow
import mlflow.spark

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress py4j verbose logging
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)
logging.getLogger("py4j.clientserver").setLevel(logging.ERROR)


def validate_and_clean_data(df):
    """Validate and clean the input data"""
    logger.info("Validating and cleaning data...")

    # Initial data info
    initial_count = df.count()
    logger.info(f"Initial dataset size: {initial_count} rows")

    # Check for null values before cleaning
    null_counts = df.select(
        [count(when(col(c).isNull(), c)).alias(c) for c in df.columns],
    ).collect()[0]
    for col_name, null_count in null_counts.asDict().items():
        if null_count > 0:
            logger.info(
                f"Column {col_name} has {null_count} null values ({null_count / initial_count * 100:.2f}%)",
            )

    # Handle missing values
    df_clean = df.dropna()
    after_dropna = df_clean.count()
    logger.info(
        f"After dropping nulls: {after_dropna} rows ({(initial_count - after_dropna) / initial_count * 100:.2f}% removed)",
    )

    # Filter out invalid data
    df_clean = df_clean.filter(
        (col("Steps") >= 0)
        & (col("Heart_Rate_avg") > 0)
        & (col("Calories_Burned") >= 0),
    )

    final_count = df_clean.count()
    logger.info(
        f"After filtering invalid data: {final_count} rows ({(after_dropna - final_count) / after_dropna * 100:.2f}% removed)",
    )

    # Show data quality summary
    logger.info("Data quality summary:")
    df_clean.describe().show()

    # Check class distribution
    logger.info("Workout type distribution:")
    class_dist = (
        df_clean.groupBy("Workout_Type").count().orderBy("count", ascending=False)
    )
    class_dist.show()

    return df_clean


def create_additional_features(df):
    """Create additional engineered features to improve model performance"""
    from pyspark.sql.functions import when

    logger.info("Creating additional features...")

    # BMI-like metric (if you had height/weight, but using calories/steps as proxy)
    df = df.withColumn("calories_per_step", col("Calories_Burned") / (col("Steps") + 1))

    # Heart rate intensity zones
    df = df.withColumn(
        "hr_zone",
        when(col("Heart_Rate_avg") < 100, "Low")
        .when(col("Heart_Rate_avg") < 140, "Moderate")
        .when(col("Heart_Rate_avg") < 170, "High")
        .otherwise("Maximum"),
    )

    # Activity intensity score
    df = df.withColumn(
        "intensity_score",
        (col("Heart_Rate_avg") * col("Calories_Burned")) / (col("Steps") + 1),
    )

    # Steps categories
    df = df.withColumn(
        "steps_category",
        when(col("Steps") < 5000, "Sedentary")
        .when(col("Steps") < 10000, "Low_Active")
        .when(col("Steps") < 15000, "Active")
        .otherwise("Very_Active"),
    )

    # Calories per heart rate (efficiency metric)
    df = df.withColumn(
        "calorie_efficiency", col("Calories_Burned") / (col("Heart_Rate_avg") + 1),
    )

    return df


def get_enhanced_feature_columns():
    """Get the enhanced feature column list"""
    base_features = ["Steps", "Heart_Rate_avg", "Calories_Burned"]
    engineered_features = ["calories_per_step", "intensity_score", "calorie_efficiency"]
    categorical_features = ["hr_zone", "steps_category"]

    return base_features, engineered_features, categorical_features


def create_preprocessing_pipeline(
        base_features: List[str],
        engineered_features: List[str],
        categorical_features: List[str],
        target_col: str,
        scale_features: bool = True,
) -> List:
    """Create preprocessing pipeline with all feature types"""
    stages = []

    # Index label column
    label_indexer = StringIndexer(
        inputCol=target_col, outputCol="label", handleInvalid="skip",
    )
    stages.append(label_indexer)

    # Index categorical features
    categorical_indexed = []
    for cat_feature in categorical_features:
        indexer = StringIndexer(
            inputCol=cat_feature,
            outputCol=f"{cat_feature}_indexed",
            handleInvalid="skip",
        )
        stages.append(indexer)
        categorical_indexed.append(f"{cat_feature}_indexed")

    # Combine all feature columns
    all_feature_cols = base_features + engineered_features + categorical_indexed

    # Assemble features
    assembler = VectorAssembler(
        inputCols=all_feature_cols,
        outputCol="raw_features" if scale_features else "features",
        handleInvalid="skip",
    )
    stages.append(assembler)

    # Optional feature scaling
    if scale_features:
        scaler = StandardScaler(
            inputCol="raw_features", outputCol="features", withStd=True, withMean=True,
        )
        stages.append(scaler)

    return stages


def evaluate_model(predictions, num_classes) -> Dict[str, float]:
    """Comprehensive model evaluation"""
    metrics = {}

    # Accuracy
    accuracy_evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy",
    )
    metrics["accuracy"] = accuracy_evaluator.evaluate(predictions)

    # F1 Score
    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1",
    )
    metrics["f1_score"] = f1_evaluator.evaluate(predictions)

    # Precision
    precision_evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="weightedPrecision",
    )
    metrics["precision"] = precision_evaluator.evaluate(predictions)

    # Recall
    recall_evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="weightedRecall",
    )
    metrics["recall"] = recall_evaluator.evaluate(predictions)

    # AUC for binary classification
    if num_classes == 2:
        auc_evaluator = BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC",
        )
        metrics["auc_roc"] = auc_evaluator.evaluate(predictions)

    return metrics


def create_confusion_matrix_plot(predictions, label_map):
    """Create confusion matrix visualization"""
    try:
        # Convert to pandas for easier plotting
        pred_df = predictions.select("label", "prediction").toPandas()

        # Map numeric labels back to strings
        pred_df["actual_label"] = pred_df["label"].map(label_map)
        pred_df["predicted_label"] = pred_df["prediction"].map(label_map)

        # Create confusion matrix
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(pred_df["actual_label"], pred_df["predicted_label"])

        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=list(label_map.values()),
            yticklabels=list(label_map.values()),
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
        plt.close()

        return "confusion_matrix.png"
    except Exception as e:
        logger.error(f"Error creating confusion matrix: {e!s}")
        return None


def create_feature_importance_plot(model, feature_cols):
    """Create feature importance visualization"""
    try:
        # Get feature importances from Random Forest
        rf_model = model.stages[-1]  # Last stage is the RF classifier
        importances = rf_model.featureImportances.toArray()

        # Create DataFrame for plotting
        importance_df = pd.DataFrame(
            {"feature": feature_cols, "importance": importances},
        ).sort_values("importance", ascending=True)

        # Plot
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df["feature"], importance_df["importance"])
        plt.title("Feature Importance (Random Forest)")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=300, bbox_inches="tight")
        plt.close()

        return "feature_importance.png"
    except Exception as e:
        logger.error(f"Error creating feature importance plot: {e!s}")
        return None


def perform_hyperparameter_tuning(df_enhanced, base_features, engineered_features, categorical_features, target_col):
    """Perform basic hyperparameter tuning for Random Forest"""
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

    logger.info("Starting hyperparameter tuning...")

    # Create preprocessing pipeline
    preprocessing_stages = create_preprocessing_pipeline(
        base_features, engineered_features, categorical_features, target_col, True,
    )

    # Define Random Forest
    rf = RandomForestClassifier(featuresCol="features", labelCol="label", seed=42)

    # Create pipeline
    pipeline = Pipeline(stages=preprocessing_stages + [rf])

    # Parameter grid for tuning
    paramGrid = (ParamGridBuilder()
                 .addGrid(rf.numTrees, [100, 200])
                 .addGrid(rf.maxDepth, [10, 15])
                 .addGrid(rf.subsamplingRate, [0.8, 0.9])
                 .build())

    # Cross validator
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1",
    )

    crossval = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=3,
        seed=42,
    )

    # Fit cross validator
    cv_model = crossval.fit(df_enhanced)

    # Get best parameters
    best_model = cv_model.bestModel
    best_rf = best_model.stages[-1]

    logger.info("Best parameters found:")
    logger.info(f"  numTrees: {best_rf.getNumTrees()}")
    logger.info(f"  maxDepth: {best_rf.getMaxDepth()}")
    logger.info(f"  subsamplingRate: {best_rf.getSubsamplingRate()}")

    return cv_model, best_rf.extractParamMap()


def main():
    """Main training pipeline"""

    # Initialize Spark (assuming it's already available)
    # from pyspark.sql import SparkSession
    # spark = SparkSession.builder.appName("FitnessTrackerML").getOrCreate()

    # Configuration
    FEATURE_COLS = ["Steps", "Heart_Rate_avg", "Calories_Burned"]
    TARGET_COL = "Workout_Type"
    TEST_SIZE = 0.2
    RANDOM_SEED = 42
    SCALE_FEATURES = True
    ENABLE_HYPERPARAMETER_TUNING = False  # Set to True to enable CV tuning

    # Random Forest parameters - tuned for better performance
    RF_PARAMS = {
        "numTrees": 200,  # Increased for better ensemble
        "maxDepth": 15,  # Deeper trees for complex patterns
        "minInstancesPerNode": 1,  # Allow more granular splits
        "subsamplingRate": 0.9,  # Use more data per tree
        "featureSubsetStrategy": "sqrt",  # Better for classification
        "maxBins": 64,  # More bins for better continuous feature handling
    }

    try:
        # Load and validate data
        logger.info("Loading data...")
        df = spark.table("bronze.fitness_tracker_data").select(
            "Steps", "Heart_Rate_avg", "Calories_Burned", "Workout_Type",
        )

        df_clean = validate_and_clean_data(df)

        # Create additional features
        df_enhanced = create_additional_features(df_clean)

        # Get feature column lists
        base_features, engineered_features, categorical_features = (
            get_enhanced_feature_columns()
        )
        all_feature_names = (
                base_features
                + engineered_features
                + [f"{cat}_indexed" for cat in categorical_features]
        )

        # Optional hyperparameter tuning
        if ENABLE_HYPERPARAMETER_TUNING:
            logger.info("Performing hyperparameter tuning...")
            cv_model, best_params = perform_hyperparameter_tuning(
                df_enhanced, base_features, engineered_features, categorical_features, TARGET_COL,
            )
            # Update RF_PARAMS with best found parameters
            for param, value in best_params.items():
                if hasattr(param, "name"):
                    RF_PARAMS[param.name] = value

        # Start MLflow run
        with mlflow.start_run() as run:
            logger.info(f"Started MLflow run: {run.info.run_id}")

            # Log parameters
            mlflow.log_param("base_features", base_features)
            mlflow.log_param("engineered_features", engineered_features)
            mlflow.log_param("categorical_features", categorical_features)
            mlflow.log_param("total_features", len(all_feature_names))
            mlflow.log_param("target_column", TARGET_COL)
            mlflow.log_param("test_size", TEST_SIZE)
            mlflow.log_param("random_seed", RANDOM_SEED)
            mlflow.log_param("scale_features", SCALE_FEATURES)
            mlflow.log_param("hyperparameter_tuning", ENABLE_HYPERPARAMETER_TUNING)

            for param, value in RF_PARAMS.items():
                mlflow.log_param(f"rf_{param}", value)

            # Split data
            logger.info("Splitting data...")
            train_df, test_df = df_enhanced.randomSplit(
                [1 - TEST_SIZE, TEST_SIZE], seed=RANDOM_SEED,
            )

            train_count = train_df.count()
            test_count = test_df.count()
            logger.info(f"Training samples: {train_count}, Test samples: {test_count}")

            mlflow.log_param("train_samples", train_count)
            mlflow.log_param("test_samples", test_count)

            # Create preprocessing pipeline
            preprocessing_stages = create_preprocessing_pipeline(
                base_features,
                engineered_features,
                categorical_features,
                TARGET_COL,
                SCALE_FEATURES,
            )

            # Define Random Forest classifier
            rf = RandomForestClassifier(
                featuresCol="features", labelCol="label", seed=RANDOM_SEED, **RF_PARAMS,
            )

            # Create complete pipeline
            pipeline = Pipeline(stages=preprocessing_stages + [rf])

            # Train model
            logger.info("Training model...")
            model = pipeline.fit(train_df)

            # Get label mapping
            label_indexer = model.stages[0]
            label_map = {i: label for i, label in enumerate(label_indexer.labels)}
            num_classes = len(label_map)

            logger.info(f"Number of classes: {num_classes}")
            logger.info(f"Class labels: {list(label_map.values())}")

            mlflow.log_param("num_classes", num_classes)
            mlflow.log_param("class_labels", list(label_map.values()))

            # Create UDF to map predictions back to labels
            map_prediction_udf = udf(
                lambda x: label_map.get(int(x), "Unknown"), StringType(),
            )

            # Make predictions
            logger.info("Making predictions...")
            train_predictions = model.transform(train_df).withColumn(
                "predicted_label", map_prediction_udf(col("prediction")),
            )
            test_predictions = model.transform(test_df).withColumn(
                "predicted_label", map_prediction_udf(col("prediction")),
            )

            # Evaluate model
            train_metrics = evaluate_model(train_predictions, num_classes)
            test_metrics = evaluate_model(test_predictions, num_classes)

            # Log metrics
            for metric_name, value in train_metrics.items():
                mlflow.log_metric(f"train_{metric_name}", value)
                logger.info(f"Train {metric_name}: {value:.4f}")

            for metric_name, value in test_metrics.items():
                mlflow.log_metric(f"test_{metric_name}", value)
                logger.info(f"Test {metric_name}: {value:.4f}")

            # Check for overfitting
            accuracy_diff = train_metrics["accuracy"] - test_metrics["accuracy"]
            mlflow.log_metric("overfitting_gap", accuracy_diff)

            if accuracy_diff > 0.15:
                logger.warning(
                    f"Potential overfitting detected. Accuracy gap: {accuracy_diff:.4f}",
                )

            # Create visualizations
            logger.info("Creating visualizations...")

            # Confusion matrix
            cm_file = create_confusion_matrix_plot(test_predictions, label_map)
            if cm_file:
                mlflow.log_artifact(cm_file)

            # Feature importance
            fi_file = create_feature_importance_plot(model, all_feature_names)
            if fi_file:
                mlflow.log_artifact(fi_file)

            # Infer model signature
            signature = infer_signature(
                train_df.select(
                    all_feature_names[:3],
                ).toPandas(),  # Use first 3 features for signature
                test_predictions.select("prediction", "predicted_label").toPandas(),
            )

            # Define pip requirements to speed up model logging
            pip_requirements = [
                "pyspark>=3.0.0",
                "mlflow>=2.0.0",
                "pandas>=1.3.0",
                "numpy>=1.20.0",
                "scikit-learn>=1.0.0",
            ]

            # Log model with explicit requirements
            mlflow.spark.log_model(
                model,
                "random_forest_model",
                signature=signature,
                pip_requirements=pip_requirements,
            )

            # Display sample predictions
            logger.info("Sample predictions:")
            sample_predictions = test_predictions.select(
                "Steps",
                "Heart_Rate_avg",
                "Calories_Burned",
                "Workout_Type",
                "predicted_label",
                "prediction",
            ).limit(20)

            sample_predictions.show()

            logger.info("Training pipeline completed successfully!")
            logger.info(f"Model test accuracy: {test_metrics['accuracy']:.4f}")

            return model, test_metrics, label_map

    except Exception as e:
        logger.error(f"Training pipeline failed: {e!s}")
        raise


model, metrics, label_map = main()

# Optional: Show final results
print("\n" + "=" * 50)
print("FINAL RESULTS")
print("=" * 50)
print(f"Test Accuracy: {metrics['accuracy']:.4f}")
print(f"Test F1-Score: {metrics['f1_score']:.4f}")
print(f"Test Precision: {metrics['precision']:.4f}")
print(f"Test Recall: {metrics['recall']:.4f}")
if "auc_roc" in metrics:
    print(f"Test AUC-ROC: {metrics['auc_roc']:.4f}")
print(f"Label Mapping: {label_map}")
print("=" * 50)
