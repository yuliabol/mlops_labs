# Exploratory Data Analysis (EDA) - Stroke Prediction Dataset

## 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure output directory for plots exists
os.makedirs("notebooks/eda_plots", exist_ok=True)

## 2. Load Data
try:
    df = pd.read_csv('data/raw/healthcare-dataset-stroke-data.csv')
    print("Data loaded successfully.")
    print(df.head())
except FileNotFoundError:
    print("Error: data/raw/healthcare-dataset-stroke-data.csv not found.")
    exit(1)

## 3. Data Inspection
print("\n--- Data Info ---")
df.info()

print("\n--- Missing Values ---")
print(df.isnull().sum())

## 4. Target Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='stroke', data=df)
plt.title('Distribution of Stroke (Target Variable)')
plt.savefig('notebooks/eda_plots/target_distribution.png')
print("\nTarget distribution plot saved to notebooks/eda_plots/target_distribution.png")
# plt.show() # Commented out for non-interactive environments

## 5. Correlation Matrix
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('notebooks/eda_plots/correlation_matrix.png')
print("Correlation matrix plot saved to notebooks/eda_plots/correlation_matrix.png")
# plt.show()
