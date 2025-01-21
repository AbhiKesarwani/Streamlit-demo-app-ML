import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title='Penguin Classifier App', layout='wide')

st.title('🐧 Penguin Species Classifier')
st.markdown("This app predicts the species of a penguin based on input features. The dataset used is the Palmer Archipelago (Antarctica) penguin dataset.")

# Load data
@st.cache
def load_data():
    return pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')

df = load_data()

# Display dataset
with st.expander('Dataset Overview'):
    st.dataframe(df)

# Data visualization
st.subheader('Data Visualization')
col1, col2 = st.columns(2)

with col1:
    st.write('### Pairplot')
    sns.pairplot(df, hue='species', height=2.5)
    st.pyplot()
    
with st.expander('Data visualization'):
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='bill_length_mm', y='body_mass_g', hue='species', ax=ax)
    st.pyplot(fig)

# Sidebar for user input
st.sidebar.header('Input Features')

island = st.sidebar.selectbox('Island', df['island'].unique())
bill_length_mm = st.sidebar.slider('Bill Length (mm)', min_value=32.1, max_value=59.6, value=43.9)
bill_depth_mm = st.sidebar.slider('Bill Depth (mm)', min_value=13.1, max_value=21.5, value=17.2)
flipper_length_mm = st.sidebar.slider('Flipper Length (mm)', min_value=172.0, max_value=231.0, value=201.0)
body_mass_g = st.sidebar.slider('Body Mass (g)', min_value=2700.0, max_value=6300.0, value=4207.0)
gender = st.sidebar.selectbox('Gender', df['sex'].unique())

# Prepare input data
input_data = pd.DataFrame({
    'island': [island],
    'bill_length_mm': [bill_length_mm],
    'bill_depth_mm': [bill_depth_mm],
    'flipper_length_mm': [flipper_length_mm],
    'body_mass_g': [body_mass_g],
    'sex': [gender]
})

# Encode categorical variables
df_encoded = pd.get_dummies(df, columns=['island', 'sex'])
input_encoded = pd.get_dummies(input_data, columns=['island', 'sex'])
input_encoded = input_encoded.reindex(columns=df_encoded.columns, fill_value=0).drop('species', axis=1)

# Model training
X = df_encoded.drop('species', axis=1)
y = df['species']
clf = RandomForestClassifier()
clf.fit(X, y)

# Prediction
prediction = clf.predict(input_encoded)
prediction_proba = clf.predict_proba(input_encoded)

# Display prediction
st.subheader('Prediction')
st.write(f'The predicted species is **{prediction[0]}**.')

st.write('### Prediction Probabilities')
proba_df = pd.DataFrame(prediction_proba, columns=clf.classes_)
st.bar_chart(proba_df.T)

st.subheader('Model Explanation')
st.markdown("""
The RandomForestClassifier is a robust machine learning algorithm that builds multiple decision trees and merges them together to get a more accurate and stable prediction.
""")

