import pandas as pd
from data_preprocessing_lib_byarat import MissingValueHandler, OutlierHandler, Scaler, TextCleaner, FeatureEngineer, DataTypeConverter, CategoricalEncoder, DateTimeHandler

# Example usage
df = pd.read_csv('your_dataset.csv')

# Handle missing values
df = MissingValueHandler.impute_with_mean(df, 'column_name')

# Handle outliers

df = OutlierHandler.iqr_outlier_detection(df, 'column_name')

# Scale data
df = Scaler.min_max_scale(df, 'column_name')

# Clean text
df = TextCleaner.clean_column(df, 'text_column')

# Feature engineering
df = FeatureEngineer.normalize_budget_by_year(df, 'budget_column', 'date_column')

# Convert data types
df = DataTypeConverter.to_numeric(df, 'column_name')

# Encode categorical data
df = CategoricalEncoder.one_hot_encode(df, 'category_column')

# Handle date and time
df = DateTimeHandler.extract_date_parts(df, 'date_column')