from django.http import HttpResponse
from django.shortcuts import render,redirect
from django.db import connection
import psycopg2
from .models import Questions
from django.contrib import messages

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
    return render(request,"test.html",{'quest':s})
def tweets_pred(request):
    handle=request.GET["handle"]
    print(handle)
    return render(request , "tweets_result.html")

        
        