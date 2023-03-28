from django.urls import path
from . import views
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
urlpatterns = [
    path('',views.index,name='index'),
    path('index',views.index,name='index'),
    path('home',views.home),
    path('test',views.test),
    path("tweets",views.tweets_pred),
    path('twitter',views.twitter),
    path('submits',views.submits),
    path('profile',views.profile),
    path('res',views.result),
    path("register",views.register,name="register"),
    
    path("login",views.login,name="login"),
    path("logout",views.logout,name="logout"),
]


urlpatterns += staticfiles_urlpatterns()