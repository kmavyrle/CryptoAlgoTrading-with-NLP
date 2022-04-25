# CryptoAlgoTrading-with-NLP
Algorithmic crypto currency trading program using natural language processing

## Introduction
As crypto sentiments play an outsized role in driving cryptocurrencies prices, this report serves to explore the possibility of using machine learning to identify global sentiments on cryptocurrencies and forecast changes in cryptocurrency prices. The large amount of crypto-related data on social networks, which is constantly growing, may prove to be useful for inferring sentiments and therefore allowing for the creation of cryptocurrencies trading strategies.

## Proposed Solution
We will be using Natural Language Processing (NLP) to conduct sentiment analysis across social media platforms and determine the sentiment on different cryptocurrencies. Through the application of machine learning, large amount of data on social media platforms can be combed through quickly and converted into information on general market sentiment for price movement predictions.

## End-to-end Machine Learning
To implement this project, we will first collect submission post titles from Reddit, a popular social network platform for cryptocurrencies. We will then pre-process the text collected with Natural Language Processing, before fitting the pre-processed text with a supervised learning model to predict the direction of price for $BTC/USD. This will then allow us to formulate a machine learning algorithm that can generate buy/sell signals for Bitcoin daily.
The collection of data and machine learning models are coded in Python. The collection of data from Reddit is done through PushShift.io Reddit API, an external API build for scrapping Reddit posts. Daily prices on Bitcoin are fetched from Yahoo Finance via the yfinance, which is a third-party Python Library that is built on top of the official Yahoo! Finance's API. 
We used Natural Language Toolkit (NLTK) to implement our NLP model and SciKit-Learn (SKLearn) to implement Logistic Regression models, Support Vector Machine models and Multinomial Naive Bayes models.
The final output of our entire machine learning efforts would be predictions on the price actions of Bitcoin. It will classify a particular day’s Reddit Submissions as 1 or 0, with 1 being a positive post and thus upward price action for bitcoin. The features that will be used for the machine learning model would be the bag of words model, where each unique word in the collection of texts forms a feature. The dependent variable is a binary class for the upward/ downward price movement of bitcoin.

## Data Collection: Web Scrapping
In total, 2 different data sets were collected for the project. First, the daily prices of Bitcoin were obtained through yfinance. We initially wanted to fetch the daily prices of other cryptocurrencies like Ether, Solana, Dogecoin and Shiba Inu. However, we felt that these cryptocurrencies did not have enough past daily returns for us to train our machine learning model with. As shown in Fig.1 below, the daily price data collected on Bitcoin started back in 2014. Meanwhile, data on Ether and Dogecoin were available only after late 2017, and data on Solana and Shiba Inu only after 2020. Therefore, we chose to use only Bitcoin for our project.

Logarithmic Graph of Daily Prices of Cryptocurrencies:
![image](https://user-images.githubusercontent.com/85161103/163224171-b11bce7e-394c-42b8-ab08-883613f37d85.png)

After retrieving the daily price data on Bitcoin, we calculated the forward-looking 1-day rolling returns. For example, the rolling return on 1st January 2020 would be calculated by averaging the price returns from 2nd January 2020 to 8th January 2020. The data set is then cleaned to remove any empty spaces and infinity.
Second, Reddit submission post titles were fetched using the PushShift.io Reddit API. By searching for the name of the cryptocurrency in their respective Subreddits (official categories within Reddit), we fetched the top 100 most popular Reddit post titles for that cryptocurrency on a specific day. These top 100 Reddit posts had been voted by Reddit users to be the most popular for that day, and we assume by the wisdom of the crowd that most of these posts are created by legitimate users expressing their opinion on cryptocurrency. We looped the fetch requests to retrieve the top 100 Reddit post titles daily from late 2014 to April 2022. Next, we combined the 100 Reddit post titles into 1 single string for every day. We ended up with 1 big string of texts per day, from late 2014 to April 2022.
After obtaining both the cleaned 1-day forward-looking rolling returns data set and the combined Reddit submission post titles data set, the data sets are merged together on dates.

## Data Processing
With the Reddit submissions post titles stored in our csv documents, we processed the submission titles to produce an array of vectorized numbers representing our features for the machine learning model. The steps are outlined as follows:
1.	Convert all titles to lower case.
2.	Tokenize the texts into sentences, with each sentence being a “token”.
3.	Removing stop words that have little impact on the prediction. This step is to remove redundant features from our model.
4.	Stemming the words in sentences. Stemming would reduce the last few characters of a word, which have no impact on the meaning of the word. For example, “Laziness” to “Lazi”.
5.	Lemmatization will reduce the stemmed words to their root form to produce meaningful words. For example, “lazi” to “lazy”. 
6.	Convert the list of sentences to a string type.
Our data processing function is as shown using various supporting functions from the Python NLTK library.

![image](https://user-images.githubusercontent.com/85161103/163224227-a172f541-c3c9-40d9-9da8-d7d5f09e5ff9.png)

We import our scrapped Reddit titles from our csv files and run our data processing function iteratively to clean the data. The changes in the submission titles are as shown:
### Pre-processed and Cleaned Data
![image](https://user-images.githubusercontent.com/85161103/163238758-a595e0ae-bad8-4cab-90af-b289a02ec9c1.png)
![image](https://user-images.githubusercontent.com/85161103/163224495-185d7362-d420-42e7-b023-301ca93517a8.png)


## Vectorization
The next step would be to vectorize the data into an array of numbers for each feature. For natural language processing, a model called the Bag-of-words is used. The Bag-of-words model is a simplifying representation used to represent texts as the bag of its unique words. For this project, we scrapped over three years of data and every single unique word from each submission will be placed in this “bag-of-words".
For example, we have a snippet of text as shown:
“It was the best of times,
It was the worst of times,
It was the age of wisdom,
It was the age of foolishness.”
The bag of words representation could be presented as a list of unique words as such:
[’It’,’was’,’the’,’best’,’worst’,’age’,’of’,’times’,’wisdom’,’foolishness’]
Train-Test Split
After the data has been vectorized, we obtain our processed data which we can use to fit into our machine learning model. To split our data into a training and test set. We used about 90% of the data for training and 10% of our data to test the predictions of the model and apply them to the trading strategy.
Model Fitting and Evaluation
We performed predictions on a few models and observed the model performance. Below are the models used:
1.	Support Vector Machine
2.	Logistic Regression
3.	Multinomial Naïve Bayes Algorithm
Testing the accuracy of the predictions, we have an accuracy of 68% for the support vector machine, 61% for the Multinomial Naive Bayes Algorithm and 69% for the Logistic Regression Model. We combined the 3 models using Ensemble Methods, whereby we got the predictions of all three models. After which, we apply majority voting where we see which class was voted the most and that would be the final class
Logistic Regression	Support Vector Machine	Multinomial Naïve Bayes Algorithm

## Application of the Model
Next, we will apply our model iteratively to our scrapped submissions using the direction of $BTC/USD price as our dependent variable. We use a for loop for each day as we iterate through the data, consistently retraining the model with new data, and using the model to predict the price action of $BTC/USD for the next day. The model classifies the price action as equal to 1 or 0 if it predicts that $BTC/USD is going up or down respectively. The function used to perform the aforementioned predictions is as shown:

## Trading Strategy Implementation
The final product of the model’s predictions is an array of 1’s and 0’s for each date. The 1’s represent a prediction that the price of bitcoin will go up for the next day, while 0’s represent a prediction that the price of bitcoin will fall. For each prediction, we will be either be going long or short bitcoin with 100% of our portfolio depending on whether the prediction is a 1 or 0, with 1 being long and 0 being short. We then plotted out the compounded returns of the trading strategy. There were periods where the model outperformed a naïve buy-and-hold strategy and periods where a buy-and-hold strategy performed better. Overall, it performed fairly well, albeit for periods which it did underperform.
![image](https://user-images.githubusercontent.com/85161103/165065271-1c3cd245-d8cc-425c-94a3-79384420e9c1.png)

## Monitoring
For the monitoring step, we set an out-of-sample dataset to test the trained model for each iteration. If the accuracy of the model falls below 55%, we will not be using the prediction of the model. Instead, we will be classifying that particular day’s text as being equal to 1 which will be our default class. This is because when the class is equal to 1, our action to take in the markets would be to long $BTC/USD as we do not want to unnecessarily short the asset. This will ensure that if the model’s predictions become too inaccurate, we will not be relying on its predictions for our trading strategy.
 
## Limitations
Overall, the model is quite feasible and generally reliable. One limitation, however, is that the results may have been due to luck and the model would not outperform the traditional buy-and-hold strategy given other circumstances as it did not consistently outperform the passive strategy. One observation is that some of the Reddit submissions scrapped are low quality posts that would not make the model’s features better. As such, we could fit a decision tree before using a post to check if it is a meaningful post before adding it to our dataset.
