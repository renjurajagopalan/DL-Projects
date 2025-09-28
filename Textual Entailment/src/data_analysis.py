# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# # Basic Dataset Statistics

# %%
df = pd.read_csv("../data/sententence_data.csv")

# %%
# Display first few rows
print("First few rows of the dataset:")
df.head()

# %% [markdown]
# ### Checking for missing Values

# %%
# Check for missing values
print("\nMissing values in each column:")
df.isnull().sum()

# %% [markdown]
# ### Distribution of entailment classes AB

# %%
print("\nDistribution of entailment_AB classes:")
ab_counts = df['entailment_AB'].value_counts()
print(ab_counts)
df['entailment_AB'].value_counts(normalize=True).round(3)

# %% [markdown]
# ### Distribution of entailment classes BA

# %%
print("\nDistribution of entailment_BA classes:")
ba_counts = df['entailment_BA'].value_counts()
print(ba_counts)
df['entailment_BA'].value_counts(normalize=True).round(3)

# %% [markdown]
# ### Visualize distribution of entailment classes

# %%
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
ax=sns.barplot(x=ab_counts.index, y=ab_counts.values)
plt.title('Distribution of entailment_AB classes')
plt.xticks(rotation=45)
plt.ylabel('Count')
ax.bar_label(ax.containers[0])

plt.subplot(1, 2, 2)
ax=sns.barplot(x=ba_counts.index, y=ba_counts.values)
plt.title('Distribution of entailment_BA classes')
plt.xticks(rotation=45)
plt.ylabel('Count')
ax.bar_label(ax.containers[0])

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Distribution of seven-category classification

# %%
df['seven_category'] = df.apply(lambda row:
                                f"{row['entailment_AB']}_{row['entailment_BA']}", axis=1)

print("\nDistribution of seven-category classification:")
seven_counts = df['seven_category'].value_counts()
print(seven_counts)
df['seven_category'].value_counts(normalize=True).round(3)

# %% [markdown]
# ### Visualize seven-category distribution

# %%
plt.figure(figsize=(12, 6))
ax= sns.barplot(x=seven_counts.index, y=seven_counts.values)
plt.title('Distribution of seven-category classification')
plt.xticks(rotation=90)
plt.ylabel('Count')
ax.bar_label(ax.containers[0])
plt.tight_layout()
plt.show()


