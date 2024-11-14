import streamlit as st
import xgboost as xgb
import pandas as pd
from huggingface_hub import hf_hub_download
import itertools
from langchain_huggingface import HuggingFaceEndpoint
import os
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


##############################################################################################################################

## SANNIDHI ## START ## 

xgboostmodel_id = "Sannidhi/stress_prediction_xgboost_model"
xgboost_model = None
model_id = "meta-llama/Llama-3.2-1B-Instruct"
generator = pipeline("text-generation", model=model_id)

def get_llm_response(prompt_text, model_id="meta-llama/Llama-3.2-3B-Instruct", max_new_tokens=256, temperature=0.5):
    """Generates a response from the Hugging Face model for a given prompt text."""
    try:
        llm = HuggingFaceEndpoint(
            repo_id=model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            token=os.getenv("HF_TOKEN")
        )
        
        system_message = "Rephrase the following text without adding any comments, feedback, or suggestions. Return only the rephrased text exactly as requested."

        prompt = PromptTemplate.from_template("{system_message}\n\n{user_text}")
        
        chat = prompt | llm.bind(skip_prompt=True) | StrOutputParser(output_key='content')
        
        response = chat.invoke(input=dict(system_message=system_message, user_text=prompt_text))
        
        return response
    
    except Exception as e:
        return f"Error generating response: {e}"

def load_xgboost_model():
    global xgboost_model
    try:
        model_path = hf_hub_download(repo_id="Sannidhi/stress_prediction_xgboost_model", filename="xgboost_model.json")

        xgboost_model = xgb.Booster()
        xgboost_model.load_model(model_path)

        return True
    except Exception as e:
        st.error(f"Error loading XGBoost model from Hugging Face: {e}")
        return False

def display_predict_stress():
    st.title("Analyse Current Stress")
    st.markdown("Answer the questions below to predict your stress level.")
    
    with st.sidebar:
        go_home = st.button("Back to Home")
        if go_home:
            st.session_state.page = "home"
            
    load_xgboost_model()

    with st.form(key="stress_form"):
        stress_questions = {
            "How many fruits or vegetables do you eat every day?": ["0", "1", "2", "3", "4", "5"],
            "How many new places do you visit in an year?": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
            "How many people are very close to you?": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
            "How many people do you help achieve a better life?": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
            "With how many people do you interact with during a typical day?": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
            "How many remarkable achievements are you proud of?": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
            "How many times do you donate your time or money to good causes?": ["0", "1", "2", "3", "4", "5"],
            "How well do you complete your weekly to-do lists?": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
            "In a typical day, how many hours do you experience 'FLOW'? (Flow is defined as the mental state, in which you are fully immersed in performing an activity. You then experience a feeling of energized focus, full involvement, and enjoyment in the process of this activity)": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
            "How many steps (in thousands) do you typically walk everyday?": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
            "For how many years ahead is your life vision very clear for?": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
            "About how long do you typically sleep?": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
            "How many days of vacation do you typically lose every year?": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
            "How often do you shout or sulk at somebody?": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
            "How sufficient is your income to cover basic life expenses (1 for insufficient, 2 for sufficient)?": ["1", "2"],
            "How many recognitions have you received in your life?": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
            "How many hours do you spend every week doing what you are passionate about?": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
            "In a typical week, how many times do you have the opportunity to think about yourself?": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
            "Age (1 = 'Less than 20' 2 = '21 to 35' 3 = '36 to 50' 4 = '51 or more')": ["1", "2", "3", "4"],
            "Gender (1 = 'Female', 0 = 'Male')": ["0", "1"]
        }

        question_to_feature_map = {
            "How many fruits or vegetables do you eat every day?": "FRUITS_VEGGIES",
            "How many new places do you visit in an year?": "PLACES_VISITED",
            "How many people are very close to you?": "CORE_CIRCLE",
            "How many people do you help achieve a better life?": "SUPPORTING_OTHERS",
            "With how many people do you interact with during a typical day?": "SOCIAL_NETWORK",
            "How many remarkable achievements are you proud of?": "ACHIEVEMENT",
            "How many times do you donate your time or money to good causes?": "DONATION",
            "How well do you complete your weekly to-do lists?": "TODO_COMPLETED",
            "In a typical day, how many hours do you experience 'FLOW'? (Flow is defined as the mental state, in which you are fully immersed in performing an activity. You then experience a feeling of energized focus, full involvement, and enjoyment in the process of this activity)": "FLOW",
            "How many steps (in thousands) do you typically walk everyday?": "DAILY_STEPS",
            "For how many years ahead is your life vision very clear for?": "LIVE_VISION",
            "About how long do you typically sleep?": "SLEEP_HOURS",
            "How many days of vacation do you typically lose every year?": "LOST_VACATION",
            "How often do you shout or sulk at somebody?": "DAILY_SHOUTING",
            "How sufficient is your income to cover basic life expenses (1 for insufficient, 2 for sufficient)?": "SUFFICIENT_INCOME",
            "How many recognitions have you received in your life?": "PERSONAL_AWARDS",
            "How many hours do you spend every week doing what you are passionate about?": "TIME_FOR_PASSION",
            "In a typical week, how many times do you have the opportunity to think about yourself?": "WEEKLY_MEDITATION",
            "Age (1 = 'Less than 20' 2 = '21 to 35' 3 = '36 to 50' 4 = '51 or more')": "AGE",
            "Gender (1 = 'Female', 0 = 'Male')": "GENDER"
        }

        response_map = {str(i): i for i in range(11)}
        response_map.update({"1": 1, "2": 2})

        responses = {}
        for question, options in stress_questions.items():
            responses[question] = st.selectbox(question, options)

        submit_button = st.form_submit_button("Submit")

        if submit_button:
            feature_dict = {question_to_feature_map[q]: response_map[responses[q]] for q in stress_questions.keys()}
            feature_df = pd.DataFrame([feature_dict])

            try:
                dmatrix = xgb.DMatrix(feature_df)
                prediction = xgboost_model.predict(dmatrix)
                st.markdown(f"### Predicted Stress Level: {prediction[0]:.2f}")
                if prediction[0] <= 1:
                    st.markdown("Your stress level is within a healthy range. Keep up the good work, and aim to maintain it for continued good health!")
                else:
                    weekly_meditation_input = feature_dict["WEEKLY_MEDITATION"]
                    sleep_hours_input = feature_dict["SLEEP_HOURS"]
                    time_for_passion_input = feature_dict["TIME_FOR_PASSION"]
                    places_visited_input = feature_dict["PLACES_VISITED"]
                    daily_steps_input = feature_dict["DAILY_STEPS"]

                    weekly_meditation_upper_bound = min(10, weekly_meditation_input + 3)
                    sleep_hours_upper_bound = min(10, sleep_hours_input + 3)
                    time_for_passion_upper_bound = min(10, time_for_passion_input + 3)
                    places_visited_upper_bound = min(10, places_visited_input + 3)
                    daily_steps_upper_bound = min(10, daily_steps_input + 3)

                    weekly_meditation_range = range(weekly_meditation_input, weekly_meditation_upper_bound + 1)
                    sleep_hours_range = range(sleep_hours_input, sleep_hours_upper_bound + 1)
                    time_for_passion_range = range(time_for_passion_input, time_for_passion_upper_bound + 1)
                    places_visited_range = range(places_visited_input, places_visited_upper_bound + 1)
                    daily_steps_range = range(daily_steps_input, daily_steps_upper_bound + 1)

                    all_combinations = itertools.product(weekly_meditation_range, sleep_hours_range, time_for_passion_range, places_visited_range, daily_steps_range)

                    best_combination = None
                    min_diff = float('inf')

                    for combination in all_combinations:
                        adjusted_feature_dict = feature_dict.copy()
                        adjusted_feature_dict["WEEKLY_MEDITATION"] = combination[0]
                        adjusted_feature_dict["SLEEP_HOURS"] = combination[1]
                        adjusted_feature_dict["TIME_FOR_PASSION"] = combination[2]
                        adjusted_feature_dict["PLACES_VISITED"] = combination[3]
                        adjusted_feature_dict["DAILY_STEPS"] = combination[4]

                        adjusted_feature_df = pd.DataFrame([adjusted_feature_dict])

                        dmatrix = xgb.DMatrix(adjusted_feature_df)
                        adjusted_prediction = xgboost_model.predict(dmatrix)
                        if adjusted_prediction[0] <= 1:
                            diff = sum(abs(adjusted_feature_dict[feature] - feature_dict[feature]) for feature in adjusted_feature_dict)
                            if diff < min_diff:
                                min_diff = diff
                                best_combination = adjusted_feature_dict
                    if best_combination:
                        best_sleep = best_combination["SLEEP_HOURS"]
                        best_meditation = best_combination["WEEKLY_MEDITATION"]
                        best_passion = best_combination["TIME_FOR_PASSION"]
                        best_places = best_combination["PLACES_VISITED"]
                        best_steps = best_combination["DAILY_STEPS"]
                        best_stress_level = xgboost_model.predict(xgb.DMatrix(pd.DataFrame([best_combination])))[0]
                        
                        prompt = f"Your stress level appears a bit elevated. To help bring it to a healthier range, try getting {best_sleep} hours of sleep each night, spend around {best_passion} hours each week doing something you’re passionate about, set aside {best_meditation} hours weekly for meditation, aim for {best_steps} thousand steps a day, and plan to explore {best_places} new places this year. These small changes can make a meaningful difference and help you reach a stress level of {best_stress_level}."
                        model_response = get_llm_response(prompt)
                        if model_response:
                            st.markdown(model_response)
                        else:
                            st.markdown("Your stress seems a bit high.")
                    else:
                        prompt = f"Your stress level seems a bit high. To help bring it down, aim for up to {sleep_hours_upper_bound} hours of sleep each night, spend around {time_for_passion_upper_bound} hours each week on activities you enjoy, set aside {weekly_meditation_upper_bound} hours for meditation each week, try to reach {daily_steps_upper_bound} thousand steps daily, and plan to explore {places_visited_upper_bound} new places this year. These small adjustments can have a positive impact on your stress levels and overall well-being."    
                        model_response = get_llm_response(prompt)
                        if model_response:
                            st.markdown(model_response)
                        else:
                            st.markdown("Your stress seems a bit high.")    
            except Exception as e:
                st.error(f"Error making prediction: {e}")

## SANNIDHI ## END ##

##############################################################################################################################