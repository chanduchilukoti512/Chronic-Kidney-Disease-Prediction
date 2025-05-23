from django.shortcuts import render
import pickle
import pandas as pd
import numpy as np

# Load the trained ML model
MODEL_PATH = "C:\\Users\\chven\\OneDrive\\Documents\\Desktop\\PROJECT\\backend\\CKD_Prediction\\ml_model\\trained_model.pkl"


try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")
    print(f"Ensure the file exists at: {MODEL_PATH}")

# Views
def mainpage(request):
    return render(request, 'mainpage.html')

def more_info(request):
    return render(request, 'moreinfo.html')

def home(request):
    return render(request, 'home.html')

def Symptoms(request):
    return render(request, 'Symptoms.html')

def prevent(request):
    return render(request, 'prevent.html')


def preprocess_input_data(input_data):
    try:
        # Map categorical data to numerical values
        input_data['Bacteria (Yes/No)'] = input_data['Bacteria (Yes/No)'].map({'Yes': 1, 'No': 0}).fillna(0)
        input_data['Hypertension (Yes/No)'] = input_data['Hypertension (Yes/No)'].map({'Yes': 1, 'No': 0}).fillna(0)
        input_data['Coronary Artery Disease (Yes/No)'] = input_data['Coronary Artery Disease (Yes/No)'].map({'Yes': 1, 'No': 0}).fillna(0)
        input_data['Appetite (Good/Poor)'] = input_data['Appetite (Good/Poor)'].map({'Good': 1, 'Poor': 0}).fillna(0)

        # Convert numeric data and handle missing values
        for column in input_data.columns:
            if input_data[column].dtype in ['float64', 'int64']:
                input_data[column] = pd.to_numeric(input_data[column], errors='coerce').fillna(input_data[column].mean())
            else:
                input_data[column] = input_data[column].fillna('')
        
        # Select required features
        input_data = input_data[[
            'Age', 'Blood Pressure (mmHg)', 'Albumin', 'Sugar',
            'Red Blood Cell Count', 'Bacteria (Yes/No)', 
            'Blood Glucose Random (mg/dL)', 'Haemoglobin (g/dL)', 
            'Hypertension (Yes/No)', 'Coronary Artery Disease (Yes/No)', 
            'Appetite (Good/Poor)'
        ]]
        return input_data
    except Exception as e:
        raise ValueError(f"Preprocessing error: {e}")

def predict(request):
    if model is None:
        # Render an error message if the model is not loaded
        return render(request, 'home.html', {'error': "Model not loaded. Please contact the administrator."})

    if request.method == 'POST':
        # Gather input data from the form
        try:
            input_data = {
                'Age': float(request.POST.get('age', 0)),  # Ensure this matches the form input names
                'Blood Pressure (mmHg)': float(request.POST.get('bp', 0)),
                'Albumin': float(request.POST.get('albumin', 0)),  # Changed to float as it is numeric
                'Sugar': float(request.POST.get('sugar', 0)),  # Changed to float as it is numeric
                'Red Blood Cell Count': float(request.POST.get('rbcc', 0)),
                'Bacteria (Yes/No)': request.POST.get('bacteria', 'No'),
                'Blood Glucose Random (mg/dL)': float(request.POST.get('bgr', 0)),
                'Haemoglobin (g/dL)': float(request.POST.get('haemoglobin', 0)),
                'Hypertension (Yes/No)': request.POST.get('hypertension', 'No'),
                'Coronary Artery Disease (Yes/No)': request.POST.get('cad', 'No'),
                'Appetite (Good/Poor)': request.POST.get('appetite', 'Good')
            }

            # Convert input data into a DataFrame for preprocessing
            input_df = pd.DataFrame([input_data])
            input_df = preprocess_input_data(input_df)

            # Make the prediction
            prediction = model.predict(input_df)
            result = "Chronic Kidney Disease Positive" if prediction[0] == 1 else "No Chronic Kidney Disease"

            # Pass the result to the template
            return render(request, 'result.html', {'result': result})

        except Exception as e:
            # Handle any errors during prediction
            result = f"Prediction error: {e}"
            return render(request, 'result.html', {'result': result})
    
    # If not a POST request, render the home page
    return render(request, 'home.html')
