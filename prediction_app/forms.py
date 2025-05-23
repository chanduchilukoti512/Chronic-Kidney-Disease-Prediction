from django import forms



class PredictionForm(forms.Form):
    Age = forms.IntegerField()
    Blood_Pressure = forms.IntegerField()
    Albumin = forms.IntegerField()
    Sugar = forms.IntegerField()
    Red_Blood_Cell_Count = forms.IntegerField()
    Bacteria = forms.ChoiceField(choices=[('Yes', 'Yes'), ('No', 'No')])
    Blood_Glucose = forms.IntegerField()
    Haemoglobin = forms.FloatField()
    Hypertension = forms.ChoiceField(choices=[('Yes', 'Yes'), ('No', 'No')])
    Coronary_Artery_Disease = forms.ChoiceField(choices=[('Yes', 'Yes'), ('No', 'No')])
    Appetite = forms.ChoiceField(choices=[('Good', 'Good'), ('Poor', 'Poor')])
