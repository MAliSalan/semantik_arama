import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import openai
import os
import numpy as np
import pickle


load_dotenv()

openai.api_key = "YOUR-API-KEY"

EMBEDDING_MODEL = "text-embedding-ada-002"
embedding_cache_path = "recommendations_embeddings_cache.pkl"

def get_embedding(text, model=EMBEDDING_MODEL):
    response = openai.Embedding.create(
        input=text,
        model=model
    )
    embedding = response['data'][0]['embedding']
    return embedding

def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)

try:
    embedding_cache = pd.read_pickle(embedding_cache_path)
except FileNotFoundError:
    embedding_cache = {}

def embedding_from_string(string: str, model: str = EMBEDDING_MODEL) -> list:
    if (string, model) not in embedding_cache:
        embedding_cache[(string, model)] = get_embedding(string, model)
        with open(embedding_cache_path, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)
    return embedding_cache[(string, model)]

st.title("OpenAI Şemantik Arama")
st.write("Demo")

uploaded_file = st.file_uploader("Dosya yükle", type=['csv', 'xlsx'])
if uploaded_file is not None:
    if '.csv' in uploaded_file.name:
        df = pd.read_csv(uploaded_file, nrows=1000)
    elif '.xlsx' in uploaded_file.name:
        df = pd.read_excel(uploaded_file, nrows=1000)

    if not df.empty:
        st.dataframe(df)
    else:
        st.error("Yüklenen dosya boş. Lütfen geçerli bir dosya yükleyin.")
else:
    st.warning("Lütfen bir dosya yükleyin.")


if not df.empty:
    id_column = st.selectbox('Hangi sütunu arayacağız?', tuple(df.columns))

 
    text_input = st.text_input("Bir metin girin:")

    if text_input:
 
        embedding = embedding_from_string(text_input)

      
        text_embeddings = [embedding_from_string(str(i)) for i in df[id_column]]

       
        scores = [cosine_similarity(i, embedding) for i in text_embeddings]

        
        df['scores'] = scores

        
        st.dataframe(df.sort_values('scores', ascending=False).head(50))
