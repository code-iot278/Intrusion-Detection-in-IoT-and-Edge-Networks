import pandas as pd

# Step 1: Load your input CSV file
input_file = "/content/drive/MyDrive/Colab Notebooks/iot final/imputed_output.csv"  # Update the path if needed
df = pd.read_csv(input_file)

# Step 2: Apply One-Hot Encoding to the 'Attack_type' column only
df_encoded = pd.get_dummies(df, columns=['Attack_type'])

# Step 3: Save the result to a new CSV file
output_file = "/content/drive/MyDrive/Colab Notebooks/iot final/encoded_output.csv"
df_encoded.to_csv(output_file, index=False)

# Step 4: Print a message confirming completion
print(f"One-Hot Encoded file saved to: {output_file}")
df_encoded