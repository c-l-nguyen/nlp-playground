import json
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer


# combine all JSON files into dataframe
with open('id-email-6.json') as json_file:
    data = json.load(json_file)

text_body = data[0]["body"]["text"]
dataset = pd.DataFrame({"Text":[text_body]})
num_words = dataset["Text"].apply(lambda x: len(str(x).split(" ")))

freq = pd.Series(' '.join(dataset['Text']).split()).value_counts()[:20]
print(freq)



# set up stop words in English
stop_words = set(stopwords.words("english"))

corpus = []
print(dataset['Text'].iloc[0])
# for i in range(0, num_words.iloc[0]):

# Remove punctuations
text = re.sub('[^a-zA-Z]', ' ', dataset['Text'].iloc[0])

# Convert to lowercase
text = text.lower()

# remove tags
text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)

# remove special characters and digits
text=re.sub("(\\d|\\W)+"," ",text)

# Convert to list from string
text = text.split()

# Stemming
ps=PorterStemmer()

# Lemmatisation
lem = WordNetLemmatizer()
text = [lem.lemmatize(word) for word in text if not word in stop_words] 
text = " ".join(text)
corpus.append(text)

print(corpus)

cv=CountVectorizer(stop_words=stop_words, max_features=10000, ngram_range=(1,3))
X=cv.fit_transform(corpus)
print(list(cv.vocabulary_.keys())[:10])

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in      
                   vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                       reverse=True)
    return words_freq[:n]

top_words = get_top_n_words(corpus, n=20)
top_df = pd.DataFrame(top_words)
top_df.columns=["Word", "Freq"]

print(top_df)

query_words = ["cheating"]

def multiplier(row):
	weight = 1
	if row["Word"] in query_words:
		weight = 10
	return row["Freq"] * weight

top_df["Freq"] = top_df.apply(multiplier, axis=1)
print(top_df.sort_values("Freq", ascending=False))

