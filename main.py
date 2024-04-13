import streamlit as st
from transformers import pipeline

# Load NLP pipelines
sentiment_pipeline = pipeline('sentiment-analysis')
generation_pipeline = pipeline('text-generation')
translation_pipeline = pipeline('translation', model="Helsinki-NLP/opus-mt-fr-en")
summarization_pipeline = pipeline('summarization')
NER_pipeline = pipeline("ner", grouped_entities=True)

st.title('NLP: Tasks Using Pipeline')

user_input = st.text_area('Enter text:')
task = st.selectbox('Choose a task:', ['Sentiment Analysis', 'Text Generation', 'Translation', 'Summarization', 'Named Entity Recognition'])

if st.button('Submit'):
    if task == 'Sentiment Analysis':
        sent = sentiment_pipeline(user_input)
        st.subheader('Sentiment analysis:')
        st.write(f"{sent[0]['label']}, Score: {sent[0]['score']}")
    elif task == 'Text Generation':
        gen = generation_pipeline(user_input)
        st.subheader('Generated Text:')
        st.write(gen[0]['generated_text'])
    elif task == 'Translation':
        trans = translation_pipeline(user_input)
        st.subheader('Translation:')
        st.write(trans[0]['translation_text'])
    elif task == 'Summarization':
        summ = summarization_pipeline(user_input)
        st.subheader('Summarization:')
        st.write(summ[0]['summary_text'])
    elif task == 'Named Entity Recognition':
        ner = NER_pipeline(user_input)
        st.subheader('NER:')
        st.write(f"group entity: {ner[0]['entity_group']}, score: {ner[0]['score']}, word: {ner[0]['word']}")
