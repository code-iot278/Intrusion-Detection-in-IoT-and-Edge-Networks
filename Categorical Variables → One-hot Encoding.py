import pandas as pd

# Load the CSV file
input_file = '/content/drive/MyDrive/Colab Notebooks/iot final/imputed_output.csv'  # Replace with your file path
df = pd.read_csv(input_file)

# Display the original data
print("Original Data:")
print(df.head())

# Automatically identify categorical columns (object or category dtype)
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
print("\nCategorical Columns:")
print(categorical_cols)

# Apply one-hot encoding
df_encoded = pd.get_dummies(df, columns=categorical_cols)

# Display the encoded data
print("\nOne-Hot Encoded Data:")
print(df_encoded.head())

# Save the result to a new CSV file
output_file = '/content/drive/MyDrive/Colab Notebooks/iot final/encoded_output.csv'
df_encoded.to_csv(output_file, index=False)
print(f"\nOne-hot encoded data saved to '{output_file}'")
df_encoded