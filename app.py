from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and scaler
model = pickle.load(open('salary_prediction_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

def get_user_input():
    user_input = {}

    # Collect user input
    user_input['YearsExperience'] = float(input('Enter Years of Experience: '))

    return pd.DataFrame(user_input, index=[0])

def make_prediction(user_input):
    # Scaling the data that the user entered
    user_input_scaled = scaler.transform(user_input)
    # Making a prediction using the trained linear regression model
    prediction = model.predict(user_input_scaled)
    # Returning the prediction
    return prediction

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect user input from the form
    years_experience = float(request.form['YearsExperience'])
    
    # Prepare the input data
    user_input = pd.DataFrame({'YearsExperience': [years_experience]})
    
    # Make prediction
    prediction = make_prediction(user_input)
    
    return render_template('index.html', prediction_text=f'Predicted Salary: ${prediction[0]:.2f}')

if __name__ == "__main__":
    app.run(debug=True)
