import pandas as pd
from sklearn.impute import SimpleImputer

# Step 1: Load the CSV file into a pandas DataFrame
file_path = "C:/Users/ACER/Downloads/housing_data.csv"  # Update with your file path
df = pd.read_csv(file_path)

# Display the original DataFrame
print("Original DataFrame:")
print(df.head())  # Display the first few rows for verification

# Step 2: Identify columns with missing values (NaN)
missing_columns = df.columns[df.isnull().any()]
print("\nColumns with missing values:")
print(missing_columns)

# Step 3: Impute missing values using mean imputation
imputer = SimpleImputer(strategy='mean')
imputed_data = imputer.fit_transform(df)

# Convert the numpy array back to a DataFrame
imputed_df = pd.DataFrame(imputed_data, columns=df.columns)

# Display the imputed DataFrame
print("\nImputed DataFrame:")
print(imputed_df.head())  # Display the first few rows of the imputed DataFrame

decimal_places = 2  # Adjust this as needed
imputed_df = imputed_df.round(decimals=decimal_places)


# Step 4: Save the imputed DataFrame back to a CSV file (optional)
imputed_file_path = 'imputed_housing_data.csv'
imputed_df.to_csv(imputed_file_path, index=False)
print(f"\nImputed DataFrame saved to {imputed_file_path}")