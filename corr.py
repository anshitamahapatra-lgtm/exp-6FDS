
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = sns.load_dataset('iris')

# Display the first five rows
print(iris.head())

# Select numerical features
data = iris.drop('species', axis=1)

# Calculate correlation
corr = data.corr()

# Display correlation matrix
print(corr)

# Plot correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')

plt.title('Correlation Heatmap of Iris Dataset')
plt.show()
