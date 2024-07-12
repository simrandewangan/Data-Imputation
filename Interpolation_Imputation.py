import pandas as pd

# Load your dataset
file_path = "C:/Users/ACER/Downloads/diabetes_data.csv"
df = pd.read_csv(file_path)

# Interpolate missing values
imputed_df = df.interpolate(method='linear', limit_direction='both')

decimal_places = 2  # Adjust this as needed
imputed_df = imputed_df.round(decimals=decimal_places)


# Save the imputed dataset to a new CSV file
imputed_file_path = 'imputed_diabetes_data.csv'
imputed_df.to_csv(imputed_file_path, index=False)

print(f"Imputed data has been saved to {imputed_file_path}")