from django.db import models

# Create your models here.

class Questions(models.Model):
    num = models.IntegerField()
    ques = models.CharField(max_length=140, null=True)
    opt1 = models.CharField(max_length=140, null=True)
    opt2 = models.CharField(max_length=140, null=True)
    opt3 = models.CharField(max_length=140, null=True)