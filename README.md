# Yelp-Review-Generator

## Project Overview

This project uses data scraped from Yelp and through natural language processing and neural networks, I have developed a text generator that will create a review post based on the rating you want to give a particular establishment. The goal of this model is to create a template for you which you can tweak slightly and post the review onto your account. This will save you a lot of time as the model has been trained on thousands of reviews for each star rating, allowing this text generator to capture the general sentiment you may be feeling for the star rating you want to provide.


## Initial EDA

I wanted to see what types of reviews users found to be funny, cool, and useful. Please see below for the top reviews from my initial scrape.

<p align="center">
  <u><b> Funniest and Cool Review </b></u>
</p> 
<p align="center">
  <img src="./Images/Funny and Cool.png" title="Funny & Cool">
</p>

<p align="center">
  <u><b> Most Useful Review </b></u>
</p> 
<p align="center">
  <img src="./Images/Useful.png" title="Funny & Cool">
</p>

Because I want to run a model for each star rating, based on my rating distribution below, I will need additional datapoints for 1, 2, and 3 star ratings. As such, in addition to the scraped reviews, I will be using the Yelp Dataset they make available every year. 

<p align="center">
  <u><b> Distribution of Ratings </b></u>
</p> 
<p align="center">
  <img src="./Images/Distribution of Ratings.png" title="Scraped">
</p>


While I wanted to initially include the most funny, cool, and useful reviews from the Yelp Dataset as well, these reviews were all about Amy's Baking Company which was features on Gordon Ramsey's show "Kitchen Nightmares". As such, I am excluding them but including the distribution of the ratings from this dataset.

<p align="center">
  <u><b> Distribution of Ratings From Yelp Dataset </b></u>
</p> 
<p align="center">
  <img src="./Images/Distribution of Ratings from Dataset.png" title="Dataset">
</p>

After taking a look through my dataset, I broke out the reviews by rating and performed a sentiment analysis using VADER (Valence Aware Dictionary and Sentiment Reasoner) to see the distribution of the compound scores amongst the different ratings. As you can see below, the average compound score as per VADER increases as the ratings increase, which is in line with what you would expect as a higher rating indicates a more positive experience.

<p align="center">
  <u><b> Distribution of VADER Scores </b></u>
</p> 
<p align="center">
  <img src="./Images/Vader.png" title="Vader">
</p>

## Basic Text Generator

Using pure NLP, I want to establish a baseline for my text generator using the 30 most common words found in each star rating. For this, I used spacy to get the lemma of the words in my review dataset and also to tokenize them. Spacy is one of the newest algorithms that offer part of speech tagging for lemmatization and tokenization which is why I ended up using it. Please see below for the text generated for each star rating:

<p align="center">
  <u><b> Basic Text Generated </b></u>
</p> 
<p align="center">
  <img src="./Images/NLP Text.png" title="Text">
</p>

To go further with this and see if the text generated matched the sentiment seen in the VADER scores above, I did a sentiment analysis of the text generated. As you can see below, the sentiment of the top 30 words generated from each of the ratings is overwhelmingly positive from a VADER standpoint. You can also see that each rating share many words in common which just means all reviews share a common pattern when they are written out.

<p align="center">
  <u><b> Sentiment of Text Generated </b></u>
</p> 
<p align="center">
  <img src="./Images/NLP Sentiment.png" title="NLP Sentiment">
</p>

We will need to develop a more robust model that takes into account the average sentiment that we have seen in the NLP EDA book which is much more representative of the negative views of the lower stars and the positive views of the 4 and 5 star ratings. For this, we will use neural networks.

## Basic Neural Networks

The goal of the neural network is to generate a sequence of texts given an input sequence (based on training data). A random number generator will select a sample line from the reviews I use to train my model and will predict the word with the highest probability based on the sequence of texts. This loop will continue until the number of words the end user chooses has been reached. 

<p align="left">
  <u><b> Cleaning Text </b></u>
</p>

For reference, please refer to the LSTM Text Generator notebook.

To perform the EDA above, my reviews were in a pandas dataframe along with other information related to each review, such as sentiment, Cool, Helpful, etc. 

I extracted the text of the reviews out from the pandas by converting the text column into a list and thus being able to clean up all the reviews at once by parsing through the list.

This was a highly iterative process because of issues presented by hardware limitations. For my project, I ended up removing punctuation to keep the unique tokens to a low enough count to not blow up my RAM. The reason for this is because "word" and "word." will be considered unique tokens and thus greatly increase the number of unique tokens that I would be basing the architecture of my neural network.

<p align="center">
  <u><b> Function used to clean text </b></u>
</p> 
<p align="center">
  <img src="./Images/cleaning_reviews.png" title="cleaner">
</p>

Using the function above, I removed text issues such as '\n\n' which is a symptom of the scrape as well as all punctuation. The number of unique tokens were nearly cut in half after removing them which made it slightly easier to run the models on my personal computer due to issues experienced with Google Colab and Google Cloud Platform.

Please see below for the number of unique tokens with and without punctuation:

<p align="center">
  <u><b> Function used to clean text </b></u>
</p> 
<p align="center">
  <img src="./Images/punctuation.png" title="punctuation">
</p>

<p align="center">
  <u><b> Function used to clean text </b></u>
</p> 
<p align="center">
  <img src="./Images/no_punctuation.png" title="no_punctuation">
</p>

Once the text was tokenized, I made sequences of 5 + 1 words. So imagine a magnifying glass on words 1-5, then shifting it over to 2-6, 3-7, etc. While most common values are 50 or even 100, because my dataset consists of many individual reviews of varying sizes, I did not believe a larger value size of 50 words would be representative of my population of reviews. A smaller length of 5 would be able to capture the context of each individual review better and thus maintain consistent ideas through the entire sequence of words which would make a sentence. However, I believe my models lacked predictive powerbecause of this limitiation. 

The more inputs you provide (i.e. 50 tokens to predict the 51st token) the more data points you are giving the model to add weights to and better understand the sequence of words. With only 5 words, my initial models really lacked predictive power which resulted in an accuracy score of only 18%. 

<p align="left">
  <u><b> First Neural Network </b></u>
</p>







### Issues with Neural Networks
Running a neural network, I learned the hard way, is very computationally expensive. 