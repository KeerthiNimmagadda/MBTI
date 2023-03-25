from django.urls import path
from . import views
urlpatterns = [
    path('',views.index,name='index'),
    path('home',views.home),
    path('test',views.test),
    path("tweets",views.tweets_pred),
    path('twitter',views.twitter),
    path('submits',views.submits)
]
