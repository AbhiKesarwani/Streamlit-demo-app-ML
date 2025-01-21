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
