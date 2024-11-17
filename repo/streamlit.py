import streamlit as st
import os

from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods

url = "https://us-south.ml.cloud.ibm.com"

def get_model(model_type,max_tokens,min_tokens,decoding, stop_sequences, temperature, top_k, top_p):

    generate_params = {
        GenParams.MAX_NEW_TOKENS: max_tokens,
        GenParams.MIN_NEW_TOKENS: min_tokens,
        GenParams.DECODING_METHOD: decoding,
        GenParams.STOP_SEQUENCES: stop_sequences,
        GenParams.TEMPERATURE: temperature,
        GenParams.TOP_K: top_k,
        GenParams.TOP_P: top_p
    }

    model = Model(
        model_id=model_type,
        params=generate_params,
        credentials={
            "apikey": st.secrets["api_key"],
            "url": url
        },
        project_id=st.secrets["project_id"]
        )

    return model

def answer_questions():
    # Header and styling
    st.set_page_config(page_title="Test Watsonx.ai LLM", page_icon="🌠", layout="wide")
    st.markdown("""
    <style>
        body { font-family: 'Arial', sans-serif; }
        .title { color: #0077b6; font-size: 2.5rem; font-weight: bold; margin-bottom: 1rem; }
    </style>
    """, unsafe_allow_html=True)
    
    # Page Title
    st.markdown('<div class="title">🌠 Test Watsonx.ai Large Language Model</div>', unsafe_allow_html=True)

    # Introductory Section
    st.markdown('<div class="info-box">Welcome!</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Made by Frizzy</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Enter a question below to test IBM Watsonx.ai LLM.</div>', unsafe_allow_html=True)

    # Input Section
    user_question = st.text_input(
        label="Ask your question",
        placeholder="For example: What is IBM?",
        help="Type your question here and press Enter."
    )

# Button to clear the input field
    if st.button("Clear", key="clear_input"):
        st.experimental_rerun()

    # Simulate model loading and response generation
    if user_question.strip():
        try:
            st.info("Generating response...")
            model_type = ModelTypes.FLAN_UL2
            max_tokens = 100
            min_tokens = 0
            decoding = DecodingMethods.SAMPLE
            stop_sequences = ['.']
            temperature = 0.7
            top_k = 50
            top_p = 1

            model = get_model(model_type, max_tokens, min_tokens, decoding, stop_sequences, temperature, top_k, top_p)

            generated_response = model.generate(prompt=user_question)

            # Display response
            response_html = f"""
            <div class="response-box">
                <b>Question:</b> {user_question}<br><br>
                <b>Response:</b><br>{generated_response}
            </div>
            """
            st.markdown(response_html, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    answer_questions()

