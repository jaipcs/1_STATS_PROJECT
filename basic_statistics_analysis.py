
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("sales_data_with_discounts.csv")

# Identify numerical and categorical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

# Descriptive Statistics for Numerical Columns
print("Descriptive Statistics:")
for col in numerical_cols:
    print(f"\nColumn: {col}")
    print(f"Mean: {df[col].mean()}")
    print(f"Median: {df[col].median()}")
    print(f"Mode: {df[col].mode()[0]}")
    print(f"Standard Deviation: {df[col].std()}")

# Data Visualization - Histograms
for col in numerical_cols:
    plt.figure()
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"histogram_{col}.png")

# Data Visualization - Boxplots
for col in numerical_cols:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"boxplot_{col}.png")

# Data Visualization - Bar Chart for Categorical Columns
for col in categorical_cols:
    plt.figure()
    df[col].value_counts().plot(kind='bar')
    plt.title(f'Bar Chart of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"barchart_{col}.png")

# Standardization (Z-score normalization)
df_standardized = df.copy()
for col in numerical_cols:
    mean = df[col].mean()
    std = df[col].std()
    df_standardized[col] = (df[col] - mean) / std

# Compare before and after standardization
print("\nBefore and After Standardization (first 5 rows):")
print(df[numerical_cols].head())
print(df_standardized[numerical_cols].head())

# One-Hot Encoding of Categorical Variables
df_encoded = pd.get_dummies(df_standardized, columns=categorical_cols)

# Display part of the transformed dataset
print("\nEncoded Dataset Sample:")
print(df_encoded.head())

# Save the transformed dataset
df_encoded.to_csv("processed_sales_data.csv", index=False)
