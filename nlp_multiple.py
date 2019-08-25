import json
import pandas as pd
import re
import glob
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

query_words = ["hackathon"]

json_files = glob.glob("*.json")

dataset = pd.DataFrame({})

# combine all JSON files into dataframe
for file in json_files:
    with open(file) as filehandler:
        data = json.load(filehandler)
    text_id = data[0]["email_id"]
    # text_data = pd.Series(data[0]["body"]["text"])
    text_data = data[0]["body"]["text"]
    text_df = pd.DataFrame({"DocID":[text_id], "Text":[text_data]})
    dataset = dataset.append(text_df, ignore_index=True)

# set up stop words in English
stop_words = set(stopwords.words("english"))

# Convert to lowercase
dataset["Parsed_Text"] = dataset["Text"].str.lower()

def clean_up(text):
    # Remove punctuations
    new_text = re.sub('[^a-zA-Z]', ' ', text)
    # Remove special characters and digits
    new_text = re.sub("(\\d|\\W)+"," ",new_text)
    new_text = new_text.split()
    return new_text

dataset["Parsed_Text"] = dataset["Parsed_Text"].apply(clean_up)

# Stemming
ps=PorterStemmer()

# Lemmatisation
lem = WordNetLemmatizer()

def good_words(row):
    corpus = []
    clean_text = [lem.lemmatize(word) for word in row if not word in stop_words]
    clean_text = " ".join(clean_text)
    corpus.append(clean_text)
    return corpus

dataset["corpus"] = dataset["Parsed_Text"].apply(good_words)

def get_top_ten_words(row):
    n = 10
    vec = CountVectorizer().fit(row)
    bag_of_words = vec.transform(row)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [[word, sum_words[0, idx]] for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

dataset["Top_10_Words"] = dataset["corpus"].apply(get_top_ten_words)

def query_multiplier(row):
    new_pair = []
    for pair in row["Top_10_Words"]:
        if pair[0] in query_words:
            weight = 10
        else:
            weight = .5
        weighted_value = pair[1] * weight
        new_pair.append([pair[0],weighted_value])
        new_pair =sorted(new_pair, key = lambda x: x[1], reverse=True)
    return new_pair

dataset["New_Top_10"] = dataset.apply(query_multiplier, axis=1)

def sum_total_points(row):
    sum_points = 0
    for pair in row:
        sum_points += pair[1]
    return sum_points

dataset["Total_Query_Points"] = dataset["New_Top_10"].apply(sum_total_points)
dataset = dataset[["DocID", "Text", "Total_Query_Points"]]
dataset = dataset.sort_values("Total_Query_Points", ascending=False)
dataset.to_csv("example.csv")
