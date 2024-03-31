import os
import pandas as pd
from pycaret.classification import predict_model, load_model
import pickle
import streamlit as st


model_dir = 'best_model'
model = load_model(model_dir)

#create streamlit app
st.title('Income Prediction')

#create form for users
st.sidebar.header('User Input Features')

#load the dataset used for training
df = pd.read_csv('who_data.csv')

#define the input fields
features = df.drop('income', axis = 1)
num_features = features.select_dtypes(include = 'number').columns.tolist()
cat_features = features.select_dtypes(include = 'object').columns.tolist()

input_fields = {}
for feature in num_features: #This allows users to select or adjust numerical values
    input_fields[feature] = st.sidebar.slider(f'select{feature}',features[feature].min(),
                                                features[feature].max(), 0)

for feature in cat_features: #This allows users to select or adjust categorical values
    input_fields[feature] = st.sidebar.selectbox(f'select{feature}',features[feature].unique()) 

#Create a dataframe for the input
user_input = pd.DataFrame([input_fields])
income_group = ['<50K', '>50K']  

#Make predictions
if st.sidebar.button('Predict'):
    prediction = predict_model(model, data = user_input, raw_score=True)
    st.write(prediction)
    predict_label = prediction['predict_label'].iloc[0]
    st.write(f'The Predicted Income Group is: {income_group[predict_label]}')




