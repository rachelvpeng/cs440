import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("final.csv")

# Print dataset
print(df.head)

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
df.columns = ['type','is_E','is_S','is_T','is_J']
df.head()