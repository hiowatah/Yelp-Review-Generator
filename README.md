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