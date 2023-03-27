from django.urls import path
from . import views
urlpatterns = [
    path("register",views.register,name="register"),
    path('index',views.index),
    path("login",views.login,name="login"),
    path("logout",views.logout,name="logout"),


]
