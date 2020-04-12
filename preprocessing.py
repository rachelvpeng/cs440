import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Function to clean text data
def cleaner(text):
    stemmer = PorterStemmer()                                        # Groups words having the same stems
    text = text.replace('|||', ' ')                                  # Replaces post separators with empty space
    text = re.sub(r'\bhttps?:\/\/.*?[\r\n]*? ', 'URL ', text, flags=re.MULTILINE)  # Replace hyperlink with 'URL'
    text = text.translate(str.maketrans('', '', string.punctuation)) # Removes punctuation
    text = text.translate(str.maketrans('', '', string.digits))      # Removes digits
    text = text.lower().strip()                                      # Converts to lower case
    final_text = []
    for w in text.split():
        if w not in stop:
            final_text.append(stemmer.stem(w.strip()))
    return ' '.join(final_text)

df = pd.read_csv("mbti_types.csv")

df_working = df.copy()
df_working.head()

# Convert to binary classifiers
df_working['I-E'] = df_working['type'].map(lambda x: 'Introversion' if x[0] == 'I' else 'Extroversion')
df_working['N-S'] = df_working['type'].map(lambda x: 'Intuition' if x[1] == 'N' else 'Sensing')
df_working['T-F'] = df_working['type'].map(lambda x: 'Thinking' if x[2] == 'T' else 'Feeling')
df_working['J-P'] = df_working['type'].map(lambda x: 'Judging' if x[3] == 'J' else 'Perceiving')

df_working.head()

# Run CountVectorizer on posts column
posts = df_working['posts'].copy()
posts = posts.apply(lambda x: x.replace('|||', ' ')) # Replaces post separators with empty space
posts.head()
cv = CountVectorizer()
cv.fit_transform(posts)

# Convert posts to dataframe to see distribution of words
df_posts = pd.DataFrame(cv.transform(posts).todense(), columns=cv.get_feature_names())
df_posts.head()
df_posts.shape
print(df_posts.sum().sort_values(ascending=False).head(10)) # Print top 10 words in posts

# Create stopwords list including 16 MBTI types to avoid biasing the model
types = ['infj', 'entp', 'intp', 'intj', 'entj', 'enfj', 'infp', 'enfp', 'isfp', \
         'istp', 'isfj', 'istj', 'estp', 'esfp', 'estj', 'esfj']
stop = stopwords.words('english')

for type in types:
    stop.append(type)

stop_rev = stop    
print(stop_rev)


# Re-run CountVectorizer to exclude stopwords, allow 2-word pairs, and limit the number of columns to 1000
cv = CountVectorizer(stop_words=stop_rev, ngram_range=(1,2), max_features=1000)
cv.fit(posts)
cv.transform(posts)

cv.transform(posts).todense()
df_posts = pd.DataFrame(cv.transform(posts).todense(), 
             columns=cv.get_feature_names())
df_posts.head()

df_posts.shape
print(df_posts.sum().sort_values(ascending=False).head(10)) # Print top 10 words in posts

# Run a final CountVectorizer on posts to include cleaning preprocessor arguments
cv = CountVectorizer(preprocessor=cleaner, stop_words=stop_rev, ngram_range=(1,2), max_features=1000)
cv.fit_transform(posts)

cv.transform(posts).todense()
df_posts = pd.DataFrame(cv.transform(posts).todense(), 
             columns=cv.get_feature_names())
df_posts.head()

