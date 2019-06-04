# Yelp-Review-Generator
Using NLP and Deep Learning, I have created a text generator that types out your review for you which saves you time and let's you get to the 'Elite' status more easily!


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