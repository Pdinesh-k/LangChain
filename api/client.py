import streamlit as st
import requests

def get_gemini_response(input_text):
    try:
        response = requests.post("http://localhost:8000/essay/invoke",
                                 json={"input": {'topic': input_text}})
        response.raise_for_status()  # Raise an HTTPError for bad responses
        return response.json().get("output", "No output key in response JSON")
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
    except ValueError:
        st.error(f"Failed to decode JSON. Response content: {response.text}")
    return None


st.title("LangChain with Gemini API")
input_text = st.text_input("Write an essay on")
input_text1 = st.text_input("Write a poem on")

if input_text:
    response = get_gemini_response(input_text)
    if response:
        st.write(response)

