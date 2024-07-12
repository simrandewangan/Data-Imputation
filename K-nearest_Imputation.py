import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = "C:/Users/ACER/Downloads/heart_data.csv"
df = pd.read_csv(file_path)

# Display initial data for debugging
print("Initial data with NaN values:")
print(df.head())

# numeric columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Initialize the KNNImputer for numeric columns
numeric_imputer = KNNImputer(n_neighbors=5)

# Impute the missing values in numeric columns
df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])

# Display data after numeric imputation for debugging
print("\nData after KNN imputation on numeric columns:")
print(df.head())

# Save the imputed dataset to a new CSV file
df.to_csv("imputed_heart_data.csv", index=False)

print("\nMissing values imputed using K-Nearest Neighbors and SimpleImputer and saved to 'imputed_heart_data.csv'")