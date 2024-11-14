## SANNIDHI ##

import streamlit as st
from homepage import display_homepage
from chatbot import display_chatbot
from predict_stress import display_predict_stress
from session_state import initialize_session_state

initialize_session_state()

if 'page' not in st.session_state:
    st.session_state.page = "home"
if st.session_state.page == "home":
    display_homepage()
elif st.session_state.page == "chat":
    display_chatbot()
elif st.session_state.page == "stress":
    display_predict_stress()

st.markdown("<style>div.stContainer {padding-top: 0;}</style>", unsafe_allow_html=True)