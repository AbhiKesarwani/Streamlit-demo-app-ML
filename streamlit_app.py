import streamlit as st

st.title('🤖 Machine Learning App')

st.info('This app builds a Machine learning model')

df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv")
df
