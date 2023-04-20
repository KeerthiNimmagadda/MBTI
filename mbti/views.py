from django.contrib import messages
from django.http import HttpResponse
from django.shortcuts import render,redirect
from django.db import connection
import psycopg2
from django.contrib.auth.models import User,auth
from .models import Questions
from django.contrib import messages
from django.http import HttpResponse
from django.shortcuts import render,redirect
from django.db import connection
import psycopg2
from .models import Questions,saveProgress
from django.contrib import messages
from django.http import HttpResponse,HttpResponseRedirect
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
from django.shortcuts import render,redirect
from django.contrib import messages
from django.contrib.auth.models import User,auth
from django.http import HttpResponse,HttpResponseRedirect
# Create your views here.

def login( request):
    response=HttpResponseRedirect('/')
    if request.method=='POST':
        username=request.POST['username']
        password=request.POST['password']
        user=auth.authenticate(username=username,password=password)
        if user is not None:
            auth.login(request,user)
            response.set_cookie('username',username)
            return response
        else:
            messages.info(request,'invalid credentials')
            return redirect('login')
    else:
        return render(request,'login.html')
    

def register(request):
    if request.method=='POST':
        first_name=request.POST['first_name']
        #last_name=request.POST['last_name']
        username=request.POST['username']
        password1=request.POST['password1']
        password2=request.POST['password2']
        #email=request.POST['email']
        email=""
        last_name=""

        if password1==password2:
            if User.objects.filter(username=username).exists():
                messages.info(request,"username taken")
                return redirect('register')
            #elif User.objects.filter(email=email).exists():
                #messages.info(request,"email taken")
                #return redirect('register')
            else:
                user=User.objects.create_user(username=username,password=password1,email=email,first_name=first_name,last_name=last_name)
                user.save()
                print('user created')
                return redirect('login')
        else:
            messages.info(request,'password not matching')
            return redirect('register')
        return redirect('/')
    else:
        return render(request,'register.html')
    
def logout(request):
    auth.logout(request)
    return redirect("/")
def index(request):
    return redirect("/")


ckey='0Cbg4uKTCBH5A04LeGotP5LF4'
csecret='oVPvWBlYNNcAA2vp0eUyD8TQd3Q2M8WaXNRxk95l7VzF69w3cm'
atoken='1314547746263625728-Elb9YP5kFu2K5RKva9FmkD13mZ9ioj'
asecret='A9NrSTJ934NhXOXxkYGYBPLc1CBRSifo9wCqjvvfBtgMc'
auth1=tweepy.OAuthHandler(ckey, csecret)
auth1.set_access_token(atoken, asecret)
api=tweepy.API(auth1)


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
	s= unidecode(s)
	POSTagger=preprocess(s)
	tweet=' '.join(POSTagger)
	stop_words = set(stopwords.words('english'))
	filtered_sentence = []
	for w in POSTagger:
	    if w not in stop_words:
	        filtered_sentence.append(w)
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
    c=request.COOKIES['username']
    qu = Questions.objects.all()
    ans= saveProgress.objects.filter(user=c).values()
    # print(ans[0]['q1'])
    # print(qu[0])
    if(len(ans)==0):
        ans=[""]*61
    else:
        ans=list(ans[0].values())
        ans=ans[2:]
    print(qu)
    print(ans)
    r=zip(qu,ans)
    return render(request,"test.html",{'que':qu,'ans':ans,'r':r});



def twitter(request):
    return render(request,"twitter.html")

def profile(request):
    c=request.COOKIES['username']
    details=User.objects.filter(username=c).values()
    l=list(details[0].values())
    print(l[5])
    progress=saveProgress.objects.filter(user=c).values()
    print("progress=",progress)
    if not progress :
        return render(request,'no_test.html',{"username":l[5]})
    mbti=list(progress[0].values())
    mbti_type=mbti[-1]
    mbti_lower=mbti_type.lower()
    print(mbti_type)
    return render(request,'profile.html',{"username":l[5],"mbti":mbti_type,"mbti_lower":mbti_lower})

def result(request):
        mbti=request.GET["mbti"]
        if mbti=="ESTJ":
            return render(request,"estj.html")
        elif mbti=="ESFJ":
            return render(request,"esfj.html")
        elif mbti=="ESFP":
            return render(request,"esfp.html")
        elif mbti=="ESTP":
            return render(request,"estp.html")
        elif mbti=="ENTJ":
            return render(request,"entj.html")
        elif mbti=="ENTP":
            return render(request,"entp.html")
        elif mbti=="ENFJ":
            return render(request,"enfj.html")
        elif mbti=="ENFP":
            return render(request,"enfp.html")
        elif mbti=="INTJ":
            return render(request,"intj.html")
        elif mbti=="INTP":
            return render(request,"intp.html")
        elif mbti=="INFJ":
            return render(request,"infj.html")
        elif mbti=="INFP":
            return render(request,"infp.html")
        elif mbti=="ISTJ":
            return render(request,"istj.html")
        elif mbti=="ISTP":
            return render(request,"istp.html")
        elif mbti=="ISFJ":
            return render(request,"isfj.html")
        elif mbti=="ISFP":
            return render(request,"isfp.html")
        return render(request,"test_result.html",{'mbti':mbti})
def submits(request):
    question=[]
    d={}
    i=1  
    username=''
    while True:
        if i==61:
            break
        s="q"+str(i)
        try:
            qt=request.GET[str(i)]
            p=qt.split('+')
            question.append(p[0])
            if p[1]=='disagree':
                d[s]='disagree'
                username=p[2]
            elif p[1]=='neutral':
                d[s]='neutral'
                username=p[2]
            elif p[1]=='agree':
                d[s]='agree'
                username=p[2]  
            
                
        except:
            d[s]=''
        i+=1
    d2={'q1': '', 'q2': '', 'q3': '', 'q4': '', 'q5': '', 'q6': '', 'q7': '', 'q8': '', 'q9': '', 'q10': '', 'q11': '', 'q12': '', 'q13': '', 'q14': '', 'q15': '', 'q16': '', 'q17': '', 'q18': '', 'q19': '', 'q20': '', 'q21': '', 'q22': '', 'q23': '', 'q24': '', 'q25': '', 'q26': '', 'q27': '', 'q28': '', 'q29': '', 'q30': '', 'q31': '', 'q32': '', 'q33': '', 'q34': '', 'q35': '', 'q36': '', 'q37': '', 'q38': '', 'q39': '', 'q40': '', 'q41': '', 'q42': '', 'q43': '', 'q44': '', 'q45': '', 'q46': '', 'q47': '', 'q48': '', 'q49': '', 'q50': '', 'q51': '', 'q52': '', 'q53': '', 'q54': '', 'q55': '', 'q56': '', 'q57': '', 'q58': '', 'q59': '', 'q60': ''}
    if d== d2:
        CRITICAL = 50
        messages.add_message(request, CRITICAL, 'PLEASE ANSWER ATLEAST ONE QUESTION')
        response=HttpResponseRedirect('/test')
        return response
    print(d)
    
    saveProgress.objects.filter(user=username).delete()
    #tweetList=request.GET["para"]
    print(question)
    tweetList=question
    #print(tweetList)
    if tweetList :
        with open('notebooks/newfrequency300.csv','rt') as f:
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
        w=saveProgress(user=username,q1=d['q1'],q2=d['q2'],q3=d['q3'],q4=d['q4'],q5=d['q5'],q6=d['q6'],q7=d['q7'],q8=d['q8'],q9=d['q9'],q10=d['q10'],q11=d['q11'],q12=d['q12'],q13=d['q13'],q14=d['q14'],q15=d['q15'],q16=d['q16'],q17=d['q17'],q18=d['q18'],q19=d['q19'],q20=d['q20'],q21=d['q21'],q22=d['q22'],q23=d['q23'],q24=d['q24'],q25=d['q25'],q26=d['q26'],q27=d['q27'],q28=d['q28'],q29=d['q29'],q30=d['q30'],q31=d['q31'],q32=d['q32'],q33=d['q33'],q34=d['q34'],q35=d['q35'],q36=d['q36'],q37=d['q37'],q38=d['q38'],q39=d['q39'],q40=d['q40'],q41=d['q41'],q42=d['q42'],q43=d['q43'],q44=d['q44'],q45=d['q45'],q46=d['q46'],q47=d['q47'],q48=d['q48'],q49=d['q49'],q50=d['q50'],q51=d['q51'],q52=d['q52'],q53=d['q53'],q54=d['q54'],q55=d['q55'],q56=d['q56'],q57=d['q57'],q58=d['q58'],q59=d['q59'],q60=d['q60'],mtype=mbti)
        w.save()
        if mbti=="ESTJ":
            return render(request,"estj.html")
        elif mbti=="ESFJ":
            return render(request,"esfj.html")
        elif mbti=="ESFP":
            return render(request,"esfp.html")
        elif mbti=="ESTP":
            return render(request,"estp.html")
        elif mbti=="ENTJ":
            return render(request,"entj.html")
        elif mbti=="ENTP":
            return render(request,"entp.html")
        elif mbti=="ENFJ":
            return render(request,"enfj.html")
        elif mbti=="ENFP":
            return render(request,"enfp.html")
        elif mbti=="INTJ":
            return render(request,"intj.html")
        elif mbti=="INTP":
            return render(request,"intp.html")
        elif mbti=="INFJ":
            return render(request,"infj.html")
        elif mbti=="INFP":
            return render(request,"infp.html")
        elif mbti=="ISTJ":
            return render(request,"istj.html")
        elif mbti=="ISTP":
            return render(request,"istp.html")
        elif mbti=="ISFJ":
            return render(request,"isfj.html")
        elif mbti=="ISFP":
            return render(request,"isfp.html")
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
            with open('notebooks/newfrequency300.csv','rt') as f:
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
        mbti="Account doesn't exist !!"
        print("no acc")
    if mbti=="ESTJ":
        return render(request,"estj.html")
    elif mbti=="ESFJ":
        return render(request,"esfj.html")
    elif mbti=="ESFP":
        return render(request,"esfp.html")
    elif mbti=="ESTP":
        return render(request,"estp.html")
    elif mbti=="ENTJ":
        return render(request,"entj.html")
    elif mbti=="ENTP":
        return render(request,"entp.html")
    elif mbti=="ENFJ":
        return render(request,"enfj.html")
    elif mbti=="ENFP":
        return render(request,"enfp.html")
    elif mbti=="INTJ":
        return render(request,"intj.html")
    elif mbti=="INTP":
        return render(request,"intp.html")
    elif mbti=="INFJ":
        return render(request,"infj.html")
    elif mbti=="INFP":
        return render(request,"infp.html")
    elif mbti=="ISTJ":
        return render(request,"istj.html")
    elif mbti=="ISTP":
        return render(request,"istp.html")
    elif mbti=="ISFJ":
        return render(request,"isfj.html")
    elif mbti=="ISFP":
        return render(request,"isfp.html")
    return render(request , "tweets_result.html" , {"mbti":mbti})

        
