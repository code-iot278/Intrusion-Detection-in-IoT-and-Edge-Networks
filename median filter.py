import pandas as pd
from sklearn.impute import SimpleImputer

# Step 1: Load the CSV file
file_path = '/content/drive/MyDrive/Colab Notebooks/iot final/iot1.csv'  # <-- Update this to your file path
df = pd.read_csv(file_path)

# Step 2: Select only numeric columns for median imputation
numeric_df = df.select_dtypes(include=['number'])

# Step 3: Apply median imputation
imputer = SimpleImputer(strategy='median')
imputed_array = imputer.fit_transform(numeric_df)

# Step 4: Convert back to DataFrame with original column names
imputed_df = pd.DataFrame(imputed_array, columns=numeric_df.columns)

# Step 5: Replace original numeric columns with imputed ones
df[numeric_df.columns] = imputed_df

# Step 6: Save the result to a new CSV file
df.to_csv('/content/drive/MyDrive/Colab Notebooks/iot final/imputed_output.csv', index=False)

print("Median imputation completed and saved to 'imputed_output.csv'")
df