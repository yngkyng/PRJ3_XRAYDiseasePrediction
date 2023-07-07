from django.db import models

class Patient(models.Model):
    idx = models.AutoField(primary_key=True)
    name = models.CharField(max_length=50, null=False)
    age = models.CharField(max_length=50, null=False)
    height = models.CharField(max_length=20, null=False)
    weight = models.CharField(max_length=50, blank=True, null=False)
    blood_type = models.CharField(max_length=20, null=True)
    last_visit = models.CharField(max_length=20, null=True)
    memo = models.CharField(max_length=500, null=True)
    picture_url = models.ImageField(null=True, max_length=150)