import numpy as np
import pickle
import streamlit

loaded_model= pickle.load(open(r'C:\Users\ROHAN\Downloads\ML Deploy\trained_model.sav','rb'))

import warnings
warnings.filterwarnings('ignore')

def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    pred = loaded_model.predict(input_data_reshaped)
    print(pred)
    if pred[0] == 0:
        return "The person is not diabetic"
    else:
        return "The person is diabetic"
    
def main():
    streamlit.title("Diabetes Predictiion Web APP")
    streamlit.header("Enter the following details to check if you are diabetic or not")
    Pregnancies = streamlit.text_input("Number of Pregnancies")
    Glucose= streamlit.text_input("Blood Glucose level")
    BloodPressure=streamlit.text_input("Blood Pressure value")
    SkinThickness=streamlit.text_input("Skin Thickness value")
    Insulin=streamlit.text_input("Insulin level")
    BMI=streamlit.text_input("BMI value")
    DiabetesPedigreeFunction=streamlit.text_input("Diabetes Pedigree Function")
    Age=streamlit.text_input("Age of the person")
    
    #code for prediction
    diagnosis=""
    
    #creating a button for prediction
    if streamlit.button("Check Diabetes"):
        diagnosis=diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
    
    streamlit.success(diagnosis)
    
    
    
if __name__=='__main__':
    main()