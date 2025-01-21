import streamlit as st
import pandas as pd

st.title('🤖 Machine Learning App')

st.info('This app builds a Machine learning model')

with st.expander('Data'):
  st.write("**Raw Data**")
  df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv")
  df

  st.write("**X**")
  g = df.drop("species", axis=1)
  g

  st.write("**y**")
  y = df.species
  y

with st.expander("Data Visualisation"):
  st.scatter_chart(data=df,x='bill_length_mm',y='body_mass_g',color='species')

with st.sidebar:
  st.header("Input Features")
  island = st.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
  bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 43.9)
  bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
  flipper_length_mm = st.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
  body_mass_g = st.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
  gender = st.selectbox('Gender', ('male', 'female'))

