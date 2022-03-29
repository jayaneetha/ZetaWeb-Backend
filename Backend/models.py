from django.db import models


class State(models.Model):
    uuid = models.CharField(max_length=40, unique=True)
    emotion = models.CharField(max_length=5)
    reward = models.DecimalField(decimal_places=2, max_digits=3, default=0)
