from django.db import models

# Create your models here.


# You can use this for future expansion, like saving user inputs or results
class Prediction(models.Model):
    age = models.IntegerField()
    blood_pressure = models.IntegerField()
    albumin = models.FloatField()
    sugar = models.FloatField()
    red_blood_cell_count = models.FloatField()
    bacteria = models.BooleanField()
    blood_glucose_random = models.FloatField()
    haemoglobin = models.FloatField()
    hypertension = models.BooleanField()
    coronary_artery_disease = models.BooleanField()
    appetite = models.BooleanField()
    prediction = models.CharField(max_length=100)  # For the predicted output

    def __str__(self):
        return f"Prediction: {self.prediction}"
