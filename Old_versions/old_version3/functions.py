import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder #Convert text into numerical form
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize #import the sentence and word tokenizer
import nltk
import datetime as dt
import time
nltk.download('stopwords')
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer 
#We will use porter stemming method
# There is also other types of stemming such as lancaster stemming
port = PorterStemmer()

from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('wordnet')#Download wordnet
wnet = WordNetLemmatizer()


def convert_to_unix(date):
  '''
  This fuction will convert date time format to traditonal unix time format
  '''
  unixd = dt.datetime.strptime(date,"%Y-%m-%d")
  return time.mktime(unixd.timetuple())

  
def convert_to_date(timestamp):
  '''
  convert unix to datetime
  '''
  return str(dt.datetime.fromtimestamp(int(timestamp)).date())

def get_data(query,
            subreddit,
            before,
            after,
            fields,
            size,
            sort_type):
    '''
    This function will get data using pushshift api from reddit.
    Maximum data size is 100 submissions
    subreddit: Which subreddit do you want to pull the submission titles from
    before: How many days before today do u want to pull
    after: How many days after do u want to pull. 
          If before = 2, and after = 100, that means you want to pull reddit titles starting from 100 days before and ending 2 days before the current day
    fields: what words do you want to find. field = bitcoin means you want to search for submissiosn containing the word [bitcoin]
    size = how many submission titles, max = 100 titles
    '''
    url = f"https://api.pushshift.io/reddit/search/submission/?q={query}&before={before}&after={after}&subreddit={subreddit}&fields={fields}&size={size}&sort_type={sort_type}"
    request = requests.get(url)
    json_response = request.json()
    return json_response['data']

def multiple_data_call(lookback_period:int,
                **kwargs):
        '''
        This is a wrapper function to wrap around the get_data function.
        Since we can only pull 100 submissions per api call
        We wrap this function to iterate through multiple times and pull HUNDRERDS AND THOUSANDS of submissions
        '''
        df = pd.DataFrame(columns = ['date','reddit_title'])
        for i in range(1,lookback_period+1):
                try:
                        after = str(i+1)+'d'
                        before = str(i)+'d'
                        response=get_data(query,subreddit,before,after,fields,size,sort_type)
                        titles = []
                        date = []         
                        for line in response:
                                date.append(convert_to_date(line['created_utc']))
                                titles.append(line['title'])
                        temp = pd.DataFrame(np.column_stack([date,titles]),columns = ['date','reddit_title'])
                        df = pd.concat([df,temp])
                except:
                        print(f"Error encountered for {convert_to_date(line['created_utc'])}")
                        continue
        df.index = list(range(len(df)))
        return df



# Data Preparation
def prepare_data(data: pd.Series)-> list:
    '''
    prepare_data function is to pre-process the scrapped reddit titles
    '''
    # 1. Convert to lower case
    print('Converting to lower case......')
    casefolded_data = data.str.strip().str.lower()

    #2 Tokenize data
    print('Tokenizing data......')
    sent_tokenized_data = list(map(word_tokenize,casefolded_data))

    #3 Remove stopwords from data
    print('Removing stopwords......')
    def remove_stopwords(sentence_list:list):
        wordlist = [word for word in sentence_list if word not in stopwords.words('english')]
        return wordlist
    stopwords_cleaned = list(map(remove_stopwords,sent_tokenized_data))

    #4 Stem words
    print('Stemming words......')
    port = PorterStemmer()
    def stem_words(wordlist:list):
        sent_list = []
        for words in wordlist:
            stemmed_w_list = [port.stem(word) for word in words]
            sent_list.append(stemmed_w_list)        
        return sent_list
    stemmed_data = stem_words(stopwords_cleaned)

    #5 wnet = WordNetLemmatizer()
    print('Lemmatizing words......')
    def lem_words(wordlist:list):
        lemmed_w_list = [wnet.lemmatize(word) for word in wordlist]
        return lemmed_w_list
    processed_data = list(map(lem_words,stemmed_data))    

    return processed_data
    

## Separated Functions
#Convert to lower case
def to_lower(df_column: pd.Series)->pd.Series:
    casefolded_data = df_column.str.strip().str.lower()
    return casefolded_data


def remove_stopwords(sentence_list:list):
    wordlist = [word for word in sentence_list if word not in stopwords.words('english')]
    return wordlist
# data['reddit_title'] = list(map(remove_stopwords,data['reddit_title']))

def stem_words(wordlist:list):
    stemmed_w_list = [port.stem(word) for word in wordlist]
    return stemmed_w_list

# data['reddit_title'] = list(map(stem_words,data['reddit_title']))

def lem_words(wordlist:list):
    lemmed_w_list = [wnet.lemmatize(word) for word in wordlist]
    return lemmed_w_list

# data['reddit_title'] = list(map(lem_words,data['reddit_title']))

def to_string(wlist:list):
    strng = ' '.join([str(item) for item in wlist])
    return strng
#for i in merged_data.index:
#   data.loc[i,'reddit_title'] = to_string(data.loc[i,'reddit_title'])

def major_vote(input):
    if input >=2:
        return 1
    else:
        return 0


    
    
