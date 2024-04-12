import streamlit as st
from transformers import pipeline

# Function to load NLP pipelines based on the selected task
def load_pipeline(task):
    if task == 'Sentiment Analysis':
        return pipeline('sentiment-analysis')
    elif task == 'Text Generation':
        return pipeline('text-generation')
    elif task == 'Translation':
        return pipeline('translation', model="Helsinki-NLP/opus-mt-fr-en")
    elif task == 'Summarization':
        return pipeline('summarization')
    elif task == 'Named Entity Recognition':
        return pipeline("ner", grouped_entities=True)
    else:
        raise ValueError(f"Invalid task: {task}")

st.title('NLP: Tasks Using Pipeline')

user_input = st.text_area('Enter text:')
task = st.selectbox('Choose a task:', ['Sentiment Analysis', 'Text Generation', 'Text Correction', 'Summarization', 'Named Entity Recognition'])

if st.button('Submit'):
    try:
        # Load the appropriate pipeline for the selected task
        nlp_pipeline = load_pipeline(task)

        # Execute the task using the loaded pipeline
        result = nlp_pipeline(user_input)

        # Display the result based on the task
        if task == 'Sentiment Analysis':
            st.subheader('Sentiment analysis:')
            st.write(f"{result[0]['label']}, Score: {result[0]['score']}")
        elif task == 'Text Generation':
            st.subheader('Generated Text:')
            st.write(result[0]['generated_text'])
        elif task == 'Translation':
            st.subheader('Translation:')
            st.write(result[0]['translation_text'])
        elif task == 'Summarization':
            st.subheader('Summarization:')
            st.write(result[0]['summary_text'])
        elif task == 'Named Entity Recognition':
            st.subheader('NER:')
            st.write(f"group entity: {result[0]['entity_group']}, score: {result[0]['score']}, word: {result[0]['word']}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
