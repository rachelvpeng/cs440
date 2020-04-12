import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk

df = pd.read_csv("mbti_types.csv")

# Print dataset
print(df.head)

# Check for null values
print(df.isnull().sum())

# Print distribution of MBTI personality types
cnt_types = df['type'].value_counts()
plt.figure(figsize=(12,4))
sns.barplot(cnt_types.index, cnt_types.values, alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('MBTI Personality Types', fontsize=12)
plt.show()

# Re-classify dataset as 4 binary classifiers
df['is_E'] = df['type'].apply(lambda x: 1 if x[0] == 'E' else 0)
df['is_S'] = df['type'].apply(lambda x: 1 if x[1] == 'S' else 0)
df['is_T'] = df['type'].apply(lambda x: 1 if x[2] == 'T' else 0)
df['is_J'] = df['type'].apply(lambda x: 1 if x[3] == 'J' else 0)
# df.columns = ['type','is_E','is_S','is_T','is_J']
print(df.head)

# Visualize I, E classifiers
labels = 'I', 'E'
sizes = [8675 - np.count_nonzero(df['is_E']), np.count_nonzero(df['is_E'])]
colors = ['gold', 'lightcoral']
explode = (0.1, 0)  # explode 1st slice
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()

# Visualize N, S classifiers
labels = 'N', 'S'
sizes = [8675 - np.count_nonzero(df['is_S']), np.count_nonzero(df['is_S'])]
colors = ['gold', 'lightcoral']
explode = (0.1, 0)  # explode 1st slice
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()

# Visualize F, T classifiers
labels = 'F', 'T'
sizes = [8675 - np.count_nonzero(df['is_T']), np.count_nonzero(df['is_T'])]
colors = ['gold', 'lightcoral']
explode = (0.1, 0)  # explode 1st slice
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()

# Visualize P, J classifiers
labels = 'P', 'J'
sizes = [8675 - np.count_nonzero(df['is_J']), np.count_nonzero(df['is_J'])]
colors = ['gold', 'lightcoral']
explode = (0.1, 0)  # explode 1st slice
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()

# Visualize correlation
sns.heatmap(df[['is_E', 'is_S', 'is_T', 'is_J']].corr(), annot = True, cmap = 'RdPu')
plt.title('Correlation of Variables')
plt.show()

