## SANNIDHI AND SRAVYA ##

import streamlit as st
from PIL import Image
import base64
from io import BytesIO

def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def display_homepage():
    image = Image.open('image.jpg')
    img_base64 = image_to_base64(image)

    st.markdown(
        f"""
        <style>
            /* Set background color to white for both light and dark mode */
            body {{
                background-color: #ffffff ;
            }}

            .image-container {{
                position: relative;
                margin-top: -70px;
                margin-left: auto;
                margin-right: auto;
                width: 300px; /* Increase width for larger image */
                display: flex;
                justify-content: center;
            }}
            .content-container {{
                margin-top: 50px; /* Adjust margin-top to make space for the image */
            }}
            .main {{
                background-color: #ffffff ;
            }}

            /* Text and button styling */
            .stButton>button {{
                background-color: #2980b9;
                color: white;
                font-size: 18px;
                font-weight: bold;
                padding: 15px 30px;
                border-radius: 8px;
                border: none;
                box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.1);
                cursor: pointer;
                transition: background-color 0.3s ease;
            }}
            .stButton>button:hover {{
                background-color: #3498db;
            }}
        </style>
        <div class="image-container">
            <img src="data:image/png;base64,{img_base64}" alt="image"/>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="content-container">
            <div style="text-align: center; font-size: 60px; font-weight: bold; color: #2c3e50;">
                AI Support for Your Mental Well-Being
            </div>
            <div style="text-align: center; font-size: 20px; margin-top: 20px; max-width: 700px; margin-left: auto; margin-right: auto;">
                Designed to streamline your mental health journey, our AI therapist is here to offer round-the-clock support, addressing mental health-related queries, providing personalized reports, and helping you understand your stress level to ensure a smoother journey toward mental and emotional health.
            </div>
        </div>
        """, unsafe_allow_html=True
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Start Chat", key="start_chat_button"):
            st.session_state.page = "chat"

    with col2:
        if st.button("Stress Test", key="predict_stress_button"):
            st.session_state.page = "stress"