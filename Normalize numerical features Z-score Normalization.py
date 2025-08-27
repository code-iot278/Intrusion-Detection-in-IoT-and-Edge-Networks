import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the CSV file
input_file = '/content/drive/MyDrive/Colab Notebooks/iot final/encoded_output.csv'
df = pd.read_csv(input_file)

# Define the label column (update the column name if needed)
label_column = 'label'  # <-- change this if your label column has a different name

# Identify numerical columns and exclude the label column
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if label_column in numerical_cols:
    numerical_cols.remove(label_column)

print("Numerical Columns (excluding label):")
print(numerical_cols)

# Apply Z-score normalization (mean = 0, std = 1)
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Display normalized data
print("\nZ-score Normalized Data (label unchanged):")
print(df.head())

# Save to new CSV
output_file = '/content/drive/MyDrive/Colab Notebooks/iot final/normalized_output.csv'
df.to_csv(output_file, index=False)
print(f"\nNormalized data saved to '{output_file}'")
df