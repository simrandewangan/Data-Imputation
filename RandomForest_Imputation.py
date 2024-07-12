import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# Load the dataset
file_path = "C:/Users/ACER/Downloads/iris_data.csv"  # Replace with your file path
df = pd.read_csv(file_path)

# Display the initial dataset
print("Initial Dataset:")
print(df.head())

# Separate features and target variable
X = df.drop(columns=['Species'])  # Features
y = df['Species']  # Target variable

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute missing values using RandomForestRegressor
imputer = SimpleImputer(strategy='mean')  # Strategy can also be 'median' or 'most_frequent'
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Display imputed datasets
print("\nImputed Training Dataset:")
print(pd.DataFrame(X_train_imputed, columns=X.columns).head())
print("\nImputed Testing Dataset:")
print(pd.DataFrame(X_test_imputed, columns=X.columns).head())