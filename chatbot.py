import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline
from langchain_huggingface import HuggingFaceEndpoint
import numpy as np
from pydub import AudioSegment
import os
from langchain.memory import ConversationBufferWindowMemory
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor

st.set_page_config(page_title="HuggingFace ChatBot", page_icon="🤗")

memory_length = 5 
memory = ConversationBufferWindowMemory(k=memory_length, memory_key="chat_history", return_messages=True)

model_id = "Sasmitah/llama_16bit_2"
model2_id = "meta-llama/Llama-3.2-3B-Instruct"
whisper_model = "openai/whisper-small" 
model1 = AutoModelForAudioClassification.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")

###################################################################################################################################

## SASMITA ## START ##
def load_transcription_model():
    try:
        transcriber = pipeline("automatic-speech-recognition", model=whisper_model)
        return transcriber
    except Exception as e:
        st.error(f"Error loading Whisper model: {e}")
        return None

def preprocess_audio(file):
    audio = AudioSegment.from_file(file).set_frame_rate(16000).set_channels(1)
    audio_samples = np.array(audio.get_array_of_samples()).astype(np.float32) / (2**15)
    return audio_samples

def transcribe_audio(file, transcriber):
    audio = preprocess_audio(file)
    transcription = transcriber(audio)["text"]
    return transcription

def predict_emotion(audio_file):                    
    if not audio_file:
        return "No audio file provided!"
    
    sound = AudioSegment.from_file(audio_file)
    sound = sound.set_frame_rate(16000)
    sound_array = np.array(sound.get_array_of_samples())

    input = feature_extractor(                                           
        raw_speech=sound_array,
        sampling_rate=16000,
        padding=True,
        return_tensors="pt")

    result = model1.forward(input.input_values.float())
    
    id2label = {
        "0": "angry",
        "1": "calm",
        "2": "disgust",
        "3": "fearful",
        "4": "happy",
        "5": "neutral",
        "6": "sad",
        "7": "surprised"
    }
    
    emotion_scores = dict(zip(id2label.values(), list(round(float(i),4) for i in result[0][0])))
    return emotion_scores    

## SASMITA ## END ##    

######################################################################################################################################################

## SRAVYA ## START ##
def get_llm_hf_inference(model_id, max_new_tokens=128, temperature=0.5):
    """Returns a language model for HuggingFace inference."""
    try:
        llm = HuggingFaceEndpoint(
            repo_id=model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            token=os.getenv("HF_TOKEN")
        )
        return llm
    except Exception as e:                                                                            
        st.error(f"Error initializing model: {e}")
        return None

def display_chatbot():

    st.title("Personal Therapist Chatbot")
    st.markdown(
    """
    🔒 *Disclaimer:* Please do not share any personal, sensitive, or confidential information during your interaction with this chatbot. This tool is for informational and supportive purposes only, and any data shared is not stored or monitored to protect your privacy.
    """
    )
    
    with st.sidebar:
        reset_history = st.button("Reset Chat History")
        go_home = st.button("Back to Home")
        if go_home:
            st.session_state.page = "home"

    if reset_history:
        st.session_state.chat_history = [{"role": "assistant", "content": st.session_state.starter_message}]
        st.session_state.user_text = None  
        st.session_state.avatars = {'user': None, 'assistant': None}  
        st.session_state.max_response_length = 1000  

    def get_response(system_message, chat_history, user_text, model_id, max_new_tokens=256):
        """Generates a response from the chatbot model."""
        hf = get_llm_hf_inference(model_id=model_id, max_new_tokens=max_new_tokens)
        if hf is None:
            return "Error: Model not initialized.", chat_history

        prompt = PromptTemplate.from_template(
            (
                "[INST] {system_message}"
                "\nCurrent Conversation:\n{chat_history}\n\n"
                
                "\nPatient: {user_text}.\n [/INST]"
                "\ntherapist:"
            )
        )
        chat = prompt | hf.bind(skip_prompt=True) | StrOutputParser(output_key='content')

        response = chat.invoke(input=dict(system_message=system_message, user_text=user_text, chat_history=chat_history))
        response = response.split("AI:")[-1].strip()

        low_engagement_threshold = 3 
        end_keywords = ["thank you", "thanks", "goodbye", "bye", "that's all"]

        short_responses = len(user_text.split()) <= low_engagement_threshold
        end_pattern_match = any(keyword in user_text.lower() for keyword in end_keywords)

        recent_short_responses = all(len(msg["content"].split()) <= low_engagement_threshold for msg in chat_history[-2:])
        response_is_acknowledgment = user_text.lower() in ["yes", "okay", "alright"]

        if (end_pattern_match or (short_responses and recent_short_responses)) and not response_is_acknowledgment:
            follow_up_question = "Would you like to have a report of your current health? Yes/No"
            response = f"I’m glad to hear that. Let’s keep checking in on this, and you can tell me how it goes next time."
            response += f"\n\n{follow_up_question}"

        chat_history.append({'role': 'user', 'content': user_text})
        chat_history.append({'role': 'assistant', 'content': response})
        return response, chat_history

    def get_summary_of_chat_history(chat_history, model2_id):
        """Generates a comprehensive summary of the chat history and a health report."""
        hf = get_llm_hf_inference(model_id=model2_id, max_new_tokens=256)
        if hf is None:
            return "Error: Model not initialized."

        chat_content = "\n".join([f"{message['role']}: {message['content']}" for message in chat_history])

        prompt = PromptTemplate.from_template(
            f"""
            Generate a detailed report based on the following conversation between a therapist and patient.
            Conversation:\n{chat_content}

            The report should include:
            1. *Patient Information:*
            - Include placeholders for Name, Age, Gender, Date of Session.

            2. *Conversation Summary:*
            - Summarize the main points of the conversation, focusing on the patient’s primary concerns and emotional state.
            - Note any specific causes of stress or distress, how these issues affect the patient's personal life, and their expressed desires or goals.

            3. *Preliminary Diagnosis:*
            - Identify the main symptoms observed in the conversation, such as mood, energy levels, motivation, etc.
            - Suggest a potential preliminary diagnosis based on the symptoms described, e.g., stress-induced burnout or other relevant concerns. Mention the need for further assessment if applicable.

            4. *Recommendations & Strategies:*
            - Provide practical, achievable strategies tailored to the patient’s needs.

            Format the report neatly with headings and subheadings as shown in the example. Aim to keep the language supportive and professional.
            """
        )

        summary = prompt | hf.bind(skip_prompt=True) | StrOutputParser(output_key='content')
        summary_response = summary.invoke(input={"chat_content": chat_content})

        return summary_response

    transcriber = load_transcription_model()
    
    input_type = st.radio("Select your input type", ("Text", "Audio"))
    
    if input_type == "Text":
        st.session_state.user_text = st.text_input("Enter your text here:")
    elif input_type == "Audio":
        audio_file = st.file_uploader("Upload an audio file for transcription", type=["mp3", "wav", "m4a"])

        if audio_file is not None and transcriber:
            with st.spinner("Transcribing audio..."):
                try:
                    st.session_state.user_text = transcribe_audio(audio_file, transcriber)
                    st.success("Audio transcribed successfully!")
                    st.audio(audio_file, format='audio/mp3')
                    emotion_result = predict_emotion(audio_file)
                    predicted_emotion = max(emotion_result, key=emotion_result.get)
                    st.write(f"Most likely emotion: {predicted_emotion.capitalize()}")
                except Exception as e:
                    st.error(f"Error transcribing audio: {e}")
                    
    

    output_container = st.container()
    
    with output_container:
        for message in st.session_state.chat_history:
            if message['role'] == 'system':
                continue
            with st.chat_message(message['role'], avatar=st.session_state['avatars'][message['role']]):
                st.markdown(message['content'])
    
    if st.session_state.user_text:
        with st.chat_message("user", avatar=st.session_state.avatars['user']):
            st.markdown(st.session_state.user_text)
    
        with st.chat_message("assistant", avatar=st.session_state.avatars['assistant']):
            response = st.session_state.chat_history[-1]['content'] if len(st.session_state.chat_history) > 2 else st.session_state.starter_message

            if "yes" in st.session_state.user_text.lower() and "Would you like to have a report of your current health? Yes/No" in response:
                with st.spinner("Generating your health report..."):
                    report = get_summary_of_chat_history(st.session_state.chat_history, model2_id)
                    st.markdown(report)
            with st.spinner("Addressing your concerns..."):
                response, st.session_state.chat_history = get_response(
                    system_message=st.session_state.system_message,
                    user_text=st.session_state.user_text,
                    chat_history=st.session_state.chat_history,
                    model_id=model_id,
                    max_new_tokens=st.session_state.max_response_length,
                )
                st.markdown(response)


## SRAVYA ## END ##

#############################################################################################################################################################################################################