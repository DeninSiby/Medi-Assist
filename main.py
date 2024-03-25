import streamlit as st
import openai
from streamlit_chat import message

import json
from feature_extractor import extraction_chain
from diagnoses_generator import generate_diagnoses

from dotenv import load_dotenv
load_dotenv()
import os

st.set_page_config(page_title="MEDI-ASSIST", page_icon="⚕")

# Set org ID and API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Sidebar
# st.sidebar.title("MEDI-ASSIST")

st.sidebar.markdown("<h1 style='text-align: center; font-size: 45px;'>MEDI-ASSIST ⚕</h1><br><br><br>", unsafe_allow_html=True)

st.sidebar.subheader("Select the Model:")

option = st.sidebar.selectbox("",["Doctor-GPT","Generate Diagnosis", "Discharge Summarizer"])

if option == "Doctor-GPT":
    
    st.markdown("<h1 style='text-align: center;'>Doctor-GPT: a totally harmless Medical Chatbot</h1>", unsafe_allow_html=True)

    # st.subheader("Doctor-GPT")
    
    # Initialise session state variables
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [
            {"role": "system", "content": """You are a medical chatbot. While I can provide information and answer questions related to the medical field, 
             it's important to note that I am not a substitute for professional medical advice. For any personal health concerns or medical advice, 
             it's always best to consult with a qualified healthcare professional"""}
        ]

    #Clear Conversation Button Implementation
    clear_button = st.button("Clear Conversation", key="clear")
    if clear_button:
        st.session_state['generated'] = []
        st.session_state['past'] = []
        st.session_state['messages'] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

    def generate_response(prompt):
        st.session_state['messages'].append({"role": "user", "content": prompt})

        completion = openai.chat.completions.create(
            model = "gpt-3.5-turbo",
            messages=st.session_state['messages']
        )
        response = completion.choices[0].message.content
        st.session_state['messages'].append({"role": "assistant", "content": response})

        return response

    # container for chat history
    response_container = st.container()
    # container for text box
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            query = st.text_area("You:", key='input', height=50)
            submit_button = st.form_submit_button(label='Generate Answer')
            
            if submit_button and query:
                output = generate_response(query)
                st.session_state['past'].append(query)
                st.session_state['generated'].append(output)
            

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))
                
    
elif option == "Generate Diagnosis":
    
    st.markdown("<h1 style='text-align: center;'>Generate Diagnosis</h1>", unsafe_allow_html=True)

    # st.subheader("Generate Diagnosis")
    # Define sex and age side by side
    col1, col2 = st.columns(2)
    with col1:
        sex = st.selectbox("Sex:", ["Male", "Female", "Other"])
    with col2:
        age = st.selectbox("Age:", ["Under 18", "18 to 30", "31 to 50", "Over 50"])

    # Text fields for symptom, past medical history, and physical examination findings
    presenting_symptom = st.text_input("Presenting Symptom")
    past_medical_history = st.text_area("Past Medical History")
    physical_exam_findings = st.text_area("Physical Examination Findings")
    
    # Creating Patient Data Dictionary
    patient_data = {
    "sex": sex,
    "age": age,
    "presenting_symptom": presenting_symptom,
    "past_medical_history": past_medical_history,
    "physical_examination_findings": physical_exam_findings
    }

    # Generate Diagnosis button
    btn_diagnosis = st.button("Generate Diagnosis")
    if btn_diagnosis:
        response = generate_diagnoses(patient_data)
        st.write(response)

elif option == "Discharge Summarizer":
    
    
    # st.subheader("Enter Discharge Summary")
    st.markdown("<h1 style='text-align: center;'>Discharge Summarizer</h1>", unsafe_allow_html=True)
    
    #Choice to type the summary or extract from file
    st.subheader("Choose Input Method:")
    input_choice = st.radio("",["Type Summary", "Upload File"])

    if input_choice == "Type Summary":
        # Discharge Summary Input:
        DS_input = st.text_area("Enter Discharge Summary Here:", height=200)

        # Generate Summary button
        btn_summary = st.button("Extract Features")

        if btn_summary:
            try:
                response = extraction_chain(DS_input)

                # Extracting information from the response
                age = response['age']
                sex = response['sex']
                presenting_symptoms = ', '.join(response['presenting_symptoms'])
                past_medical_history = ', '.join(response['past_medical_history'])
                physical_exam_findings = '\n'.join(response['physical_examination_findings'])
                major_procedures = '\n'.join(response['major_procedures'])
                medications_on_admission = '\n'.join(response['medications_on_admission'])
                medications_on_discharge = '\n'.join(response['medications_on_discharge'])
                discharge_diagnosis = '\n'.join(response['discharge_diagnosis'])
                discharge_condition = '\n'.join(response['discharge_condition'])
                discharge_instructions = '\n'.join(response['discharge_instructions'])

                # Transposing data for pivoting
                features = ["Age", "Sex", "Presenting Symptoms",
                            "Past Medical History", "Physical Examination Findings",
                            "Major Procedures", "Medications on Admission","Medications of Discharge",
                            "Discharge Diagnosis","Discharge Condition", "Discharge Instructions"]
                
                values = [age, sex, presenting_symptoms,
                        past_medical_history, physical_exam_findings, major_procedures,
                        medications_on_admission, medications_on_discharge, discharge_diagnosis,
                        discharge_condition, discharge_instructions]

                # Creating a structured table
                table_data = {feature: value for feature,
                            value in zip(features, values)}

                # Displaying pivoted table
                st.subheader("Summary:")
                st.table(table_data)

            except ValueError:
                st.error(
                    "Error occurred during feature extraction. Please check the input and try again.")
        
    elif input_choice == "Upload File":
        uploaded_file = st.file_uploader("Upload a file", type=["pdf", "doc", "txt"])

        if uploaded_file:
            file_contents = uploaded_file.read()
            text = file_contents.decode('utf-8')

            # Display uploaded text
            # st.write("Uploaded Text:")
            # st.write(text)

            # Generate Summary button
            btn_summary = st.button("Extract Features")
            
            if btn_summary:
                try:
                    response = extraction_chain(text)

                    # Extracting information from the response
                    age = response['age']
                    sex = response['sex']
                    presenting_symptoms = ', '.join(response['presenting_symptoms'])
                    past_medical_history = ', '.join(response['past_medical_history'])
                    physical_exam_findings = '\n'.join(response['physical_examination_findings'])
                    major_procedures = '\n'.join(response['major_procedures'])
                    medications_on_admission = '\n'.join(response['medications_on_admission'])
                    medications_on_discharge = '\n'.join(response['medications_on_discharge'])
                    discharge_diagnosis = '\n'.join(response['discharge_diagnosis'])
                    discharge_condition = '\n'.join(response['discharge_condition'])
                    discharge_instructions = '\n'.join(response['discharge_instructions'])

                    # Transposing data for pivoting
                    features = ["Age", "Sex", "Presenting Symptoms",
                                "Past Medical History", "Physical Examination Findings",
                                "Major Procedures", "Medications on Admission","Medications of Discharge",
                                "Discharge Diagnosis","Discharge Condition", "Discharge Instructions"]
                    
                    values = [age, sex, presenting_symptoms,
                            past_medical_history, physical_exam_findings, major_procedures,
                            medications_on_admission, medications_on_discharge, discharge_diagnosis,
                            discharge_condition, discharge_instructions]

                    # Creating a structured table
                    table_data = {feature: value for feature,
                                value in zip(features, values)}

                    # Displaying pivoted table
                    st.subheader("Summary:")
                    st.table(table_data)

                except ValueError:
                    st.error(
                        "Error occurred during feature extraction. Please check the input and try again.")