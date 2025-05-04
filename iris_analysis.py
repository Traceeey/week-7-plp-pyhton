import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset
try:
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    print("First 5 rows of the dataset:")
    print(df.head())

    print("\nData Types:")
    print(df.dtypes)

    print("\nChecking for missing values:")
    print(df.isnull().sum())

    # No missing values to clean in Iris, but let's show how to handle it
    df = df.dropna()

except FileNotFoundError:
    print("Dataset file not found.")
except Exception as e:
    print(f"An error occurred: {e}")

# Task 2: Basic Data Analysis
print("\nBasic Statistics:")
print(df.describe())

print("\nMean values grouped by species:")
print(df.groupby('species').mean())

# Task 3: Data Visualization

# Set Seaborn style
sns.set(style="whitegrid")

# 1. Line chart showing trends (simulate by plotting mean values across features)
mean_vals = df.groupby('species').mean().T
mean_vals.plot(kind='line', marker='o')
plt.title('Mean Feature Values per Species')
plt.xlabel('Feature')
plt.ylabel('Mean Value')
plt.legend(title='Species')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Bar chart showing average petal length per species
plt.figure(figsize=(6, 4))
sns.barplot(x='species', y='petal length (cm)', data=df)
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.tight_layout()
plt.show()

# 3. Histogram of sepal length
plt.figure(figsize=(6, 4))
plt.hist(df['sepal length (cm)'], bins=10, color='skyblue', edgecolor='black')
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 4. Scatter plot between sepal length and petal length
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.tight_layout()
plt.show()

# Observations
print("\nObservations:")
print("- The setosa species has significantly shorter petal lengths and widths.")
print("- Versicolor and virginica have overlapping features, but virginica tends to have larger dimensions.")
print("- Sepal length and petal length show a positive correlation.")
