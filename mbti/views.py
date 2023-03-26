from django.http import HttpResponse
from django.shortcuts import render,redirect
from django.db import connection
import psycopg2
from .models import Questions
from django.contrib import messages


from django.shortcuts import render
from django.http import HttpResponse
from joblib import load
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import *
from nltk.stem.snowball import SnowballStemmer
import tweepy
import sys
import os
import nltk 
import re
import numpy as np
import string
from unidecode import unidecode
import csv
from itertools import islice
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import nltk
nltk.download('stopwords')
import nltk
nltk.download('punkt')

ckey='0Cbg4uKTCBH5A04LeGotP5LF4'
csecret='oVPvWBlYNNcAA2vp0eUyD8TQd3Q2M8WaXNRxk95l7VzF69w3cm'
atoken='1314547746263625728-Elb9YP5kFu2K5RKva9FmkD13mZ9ioj'
asecret='A9NrSTJ934NhXOXxkYGYBPLc1CBRSifo9wCqjvvfBtgMc'
auth=tweepy.OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)
api=tweepy.API(auth)


emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)


regex_str = [
    emoticons_str,
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]

tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)


def tokenize(s):
    return tokens_re.findall(s)


def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

def preproc(s):
	#s=emoji_pattern.sub(r'', s) # no emoji
	s= unidecode(s)
	POSTagger=preprocess(s)
	#print(POSTagger)

	tweet=' '.join(POSTagger)
	stop_words = set(stopwords.words('english'))
	#word_tokens = word_tokenize(tweet)
	#filtered_sentence = [w for w in word_tokens if not w in stop_words]
	filtered_sentence = []
	for w in POSTagger:
	    if w not in stop_words:
	        filtered_sentence.append(w)
	#print(word_tokens)
	#print(filtered_sentence)
	stemmed_sentence=[]
	stemmer2 = SnowballStemmer("english", ignore_stopwords=True)
	for w in filtered_sentence:
		stemmed_sentence.append(stemmer2.stem(w))
	#print(stemmed_sentence)

	temp = ' '.join(c for c in stemmed_sentence if c not in string.punctuation) 
	preProcessed=temp.split(" ")
	final=[]
	for i in preProcessed:
		if i not in final:
			if i.isdigit():
				pass
			else:
				if 'http' not in i:
					final.append(i)
	temp1=' '.join(c for c in final)
	#print(preProcessed)
	return temp1
def getTweets(user):
	csvFile = open('user.csv', 'w', newline='')
	csvWriter = csv.writer(csvFile)
	try:
		for i in range(0,4):
			tweets=api.user_timeline(screen_name = user, count = 1000, include_rts=True)
			for status in tweets:
				tw=preproc(status.text)
				if tw.find(" ") == -1:
					tw="blank"
				csvWriter.writerow([tw])
	except AttributeError :
		print("\nFailed to run the command on that user, Skipping...")
		return -1
	except tweepy.errors.NotFound :
		print("\nUser not found")
		return -1
	except tweepy.errors.TweepyException :
		print("\nFailed to run the command on that user, Skipping...")
		return -1
	csvFile.close()

model1=load('./savedModels/model1.joblib')
model2=load('./savedModels/model2.joblib')
model3=load('./savedModels/model3.joblib')
model4=load('./savedModels/model4.joblib')
# Create your views here.
def home(request):
    return render(request,"home.html",{'name':'Keerzy'});

def index(request):
    return render(request,"index.html");

def test(request):

    qu = Questions.objects.all()
   
    return render(request,"test.html",{'que':qu});

def twitter(request):
    return render(request,"twitter.html")

def submits(request):
    question=[]
    for i in range(1,3):
        q=request.GET[str(i)]
        question.append(q)
    s=[]
    for i in question:
        s.append(i)
    print(s)
    #tweetList=request.GET["para"]
    tweetList=s
    #print(tweetList)
    if tweetList :
        with open('C:/Users/harsha vardhan/Documents/twitter/notebooks/newfrequency300.csv','rt') as f:
            csvReader=csv.reader(f)
            mydict={rows[1]: int(rows[0]) for rows in csvReader}
        vectorizer=TfidfVectorizer(vocabulary=mydict,min_df=1)
        x=vectorizer.fit_transform(tweetList).toarray()
        df=pd.DataFrame(x)
        answer=[]
        PJ=model1.predict(df)
        IE=model2.predict(df)
        TF=model3.predict(df)
        SN=model4.predict(df)
        b = Counter(IE)
        value=b.most_common(1)
        print(value)
        if value[0][0] == 1.0:
            answer.append("I")
        else:
            answer.append("E")

        b = Counter(SN)
        value=b.most_common(1)
        print(value)
        if value[0][0] == 1.0:
            answer.append("S")
        else:
            answer.append("N")

        b = Counter(TF)
        value=b.most_common(1)
        print(value)
        if value[0][0] == 1:
            answer.append("T")
        else:
            answer.append("F")

        b = Counter(PJ)
        value=b.most_common(1)
        print(value)
        if value[0][0] == 1:
            answer.append("P")
        else:
            answer.append("J")
        mbti="".join(answer)
        print(mbti)
        return render(request,"test_result.html",{'mbti':mbti})
def tweets_pred(request):
    username=request.GET["handle"]
    print(username)
    st = getTweets(username)
    print(st)
    if st != -1 :
        with open('user.csv','rt') as f:
            csvReader=csv.reader(f)
            tweetList=[rows[0] for rows in csvReader]
            #tweetList=['I prefer to completely finish one project before starting another','I like to use organizing tools like schedules and lists','Even a small mistake can cause me to doubt your overall abilities and knowledge','I feel comfortable just walking up to someone I find interesting and striking up a conversation','i are not too interested in discussing various interpretations and analyses of creative works','I am more inclined to follow my head than my heart','i usually prefer just doing what i feel like at any given moment instead of planning a particular daily routine','i rarely worry about whether i make a good impression on people i meet','i enjoy participating in group activities','i like books and movies that make i come up with my own interpretation of the ending','my happiness does not comes more from helping others accomplish things than my own accomplishments','i enjoy watching people argue i tend to avoid drawing attention to yourself my mood can change very quickly','I often make new friends','I dont waste my time learning about random topics','I am very sentimental','I like to use organizing tools like schedules and lists.','Being around lots of people energizes me.','I am always the leader wherever I go','I enjoy watching people argue']
            #print(tweetList)
        if tweetList :
            with open('C:/Users/harsha vardhan/Documents/twitter/notebooks/newfrequency300.csv','rt') as f:
                csvReader=csv.reader(f)
                mydict={rows[1]: int(rows[0]) for rows in csvReader}

            vectorizer=TfidfVectorizer(vocabulary=mydict,min_df=1)
            x=vectorizer.fit_transform(tweetList).toarray()
            df=pd.DataFrame(x)
            answer=[]
            PJ=model1.predict(df)
            IE=model2.predict(df)
            TF=model3.predict(df)
            SN=model4.predict(df)
            b = Counter(IE)
            value=b.most_common(1)
            print(value)
            if value[0][0] == 1.0:
                answer.append("I")
            else:
                answer.append("E")

            b = Counter(SN)
            value=b.most_common(1)
            print(value)
            if value[0][0] == 1.0:
                answer.append("S")
            else:
                answer.append("N")

            b = Counter(TF)
            value=b.most_common(1)
            print(value)
            if value[0][0] == 1:
                answer.append("T")
            else:
                answer.append("F")

            b = Counter(PJ)
            value=b.most_common(1)
            print(value)
            if value[0][0] == 1:
                answer.append("P")
            else:
                answer.append("J")
            mbti="".join(answer)
            print(mbti)
        else:
            mbti="\nAccount is either private or No tweets posted!"
            print("\nAccount is either private or No tweets posted!")
    else:
        mbti="no acc"
        print("no acc")
    return render(request , "tweets_result.html" , {"mbti":mbti})

        
        