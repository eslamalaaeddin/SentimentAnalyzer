# SentimentAnalyzer
Simple sentiment analyzer that classifies tweets as either positive or negative.

# Pipeline

# Data Aquistion
  From nltk.twitter_samples
  
# Splitting
	We have 10000 tweets, 5000 are Positive(+) and 5000 are Negative(-)
	We will split to be 80% for training and 20% testing

# Text Cleaning
	for each tweet
		1- Remove stock market tickers
		2- Remove old retweet text "RT"
		3- Remove hyperlinks
		4- Remove hashtags 

# PreProcessing(Ordered)
		1- Tokenization
		2- Remove stop words
		3- Remove punctuations
		4- Stemming
Output: list of processed words for each tweet.

Building Frequency Table
	for each list of words
		(word, class[+|-]): count


# Exracting Features From Tweets
	for each processed tweet
		tweet = (1, count of positive words, count of negative words)
		
Building X and Y (Real Dataset)
	X: tweet 1 (1, 10, 13) 
		 tweet 2 (1, 5, 11)
		 tweet 3 (1, 9, 18)
		 tweet 4 (1, 24, 13)
			... and so on    
	Y: 1 (positive tweet) or  0 (negative tweet)
	
# Modeling
	Logisitic Regression
	SVM 
  	Naive Bayes

# Evaluation
	Precision
	Recall
	F1 Score
