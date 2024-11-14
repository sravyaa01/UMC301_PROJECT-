## SRAVYA ##

import streamlit as st

def initialize_session_state():
    if "avatars" not in st.session_state:
        st.session_state.avatars = {'user': None, 'assistant': None}

    if 'user_text' not in st.session_state:
        st.session_state.user_text = None

    if "max_response_length" not in st.session_state:
        st.session_state.max_response_length = 2000

    if "system_message" not in st.session_state:
        st.session_state.system_message = """
        You are a compassionate therapist dedicated to deeply understanding the user's experiences and emotions. Listen carefully and respond with empathy. Reflect back what the user is expressing, validate their feelings, and ask open-ended questions that encourage self-reflection. Avoid generic advice or impersonal suggestions; instead, offer thoughtful insights that relate specifically to the user's unique challenges. Always remain in the role of a therapist, aiming to support and guide the user with patience, avoiding negative or dismissive language. Ask one question at a time. Respond in 3-4 sentences at a time.
        """

    if "starter_message" not in st.session_state:
        st.session_state.starter_message = "Hello! How are you feeling today? What are you here to talk about? I would like to get a sense of your overall life and emotional state so that I can better help you. We can start by talking about recent changes in your life or how you are feeling."

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "assistant", "content": st.session_state.starter_message}, {"role": "system", "content": st.session_state.system_message}]