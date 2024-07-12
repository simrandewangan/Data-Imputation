import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

# Load your red wine dataset CSV file into a DataFrame
file_path = "C:/Users/ACER/Downloads/winequality-red_data.csv"
df = pd.read_csv(file_path)

# Separate numeric and categorical columns (assuming all columns are numeric)
numeric_columns = df.select_dtypes(include=['number']).columns
categorical_columns = df.select_dtypes(exclude=['number']).columns

# Create separate DataFrames for numeric and categorical data
df_numeric = df[numeric_columns]
df_categorical = df[categorical_columns]

# Initialize the MICE imputer for numeric data
mice_imputer = IterativeImputer(random_state=0)

# Perform imputation on numeric data
imputed_values = mice_imputer.fit_transform(df_numeric)

# Convert the imputed values array back to a DataFrame with columns
imputed_df_numeric = pd.DataFrame(imputed_values, columns=df_numeric.columns)

# Combine imputed numeric data with original categorical data
imputed_df = pd.concat([imputed_df_numeric, df_categorical], axis=1)

decimal_places = 2  # Adjust this as needed
imputed_df = imputed_df.round(decimals=decimal_places)

# Save the imputed DataFrame to a new CSV file
output_file = 'imputed_winequality-red_data.csv'
imputed_df.to_csv(output_file, index=False)

print("Imputation completed and saved to", output_file)