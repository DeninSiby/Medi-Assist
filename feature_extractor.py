# Kor!
from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number

# LangChain Models
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain import PromptTemplate


# Standard Helpers
import pandas as pd
import requests
import time
import json
from datetime import datetime

# Text Helpers
from bs4 import BeautifulSoup
from markdownify import markdownify as md


def printOutput(output):
    print(json.dumps(output,sort_keys=True, indent=3))
    
import os

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)
    

# LLM: GPT 3.5 turbo
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    # model_name="gpt-4",
    temperature=0,
    max_tokens=4000,
    openai_api_key= os.environ["openai_api_key"]
)


# Extraction Schema for a Discharge Summary
discharge_summary_schema = Object(
    id="discharge_summary",
    description="Extracting information from a patient discharge summary.",
    attributes=[
        Text(
            id="age",
            description="Age of the patient.",
            examples=[("This is an 81-year-old female.", "81")],
            many=False,
        ),
        Text(
            id="sex",
            description="Gender of the patient.",
            examples=[("Sex: F", "F")],
            many=False,
        ),
        Text(
            id="presenting_symptoms",
            description="Symptoms presented by the patient.",
            examples=[
                ("The patient presented with three days of shortness of breath.", "shortness of breath"),
                ("Chest pressure occasionally with shortness of breath with exertion.", "chest pressure, shortness of breath")
            ],
            many=True,
        ),
        
         Text(
            id="past_medical_history",
            description="Past medical history of the patient.",
            examples=[
                ("History of emphysema (not on home O2)", "emphysema"),
                ("COPD. Last pulmonary function tests in [**2117-11-3**] demonstrated...", "COPD"),
                ("Angina: Most recent stress test was in [**2118-1-3**]...", "angina")
            ],
            many=True,
        ),

        Text(
            id="physical_examination_findings",
            description="Findings from the physical examination.",
            examples=[
                ("Blood pressure 142/76, heart rate 100 and regular", "blood pressure 142/76, heart rate 100"),
                ("Pupils are equal, round, and reactive to light and accommodation", "equal pupils, reactive to light")
            ],
            many=True,
        ),
        
        Text(
            id="major_procedures",
            description="Major procedures performed during the hospital stay.",
            examples=[
                ("The patient underwent intubation and arterial line placement.", "intubation, arterial line placement"),
                ("PICC line placement and Esophagogastroduodenoscopy were performed.", "PICC line placement, Esophagogastroduodenoscopy")
            ],
            many=True,
        ),
        Text(
            id="medications_on_admission",
            description="List of medications prescribed at admission.",
            examples=[
                ("Started on prednisone taper and oxygen at home.", "prednisone taper, oxygen at home"),
                ("On admission, the patient was taking lisinopril and atorvastatin.", "lisinopril, atorvastatin")
            ],
            many=True,
        ),
        Text(
            id="medications_on_discharge",
            description="List of medications prescribed at discharge.",
            examples=[
                ("Discharge medications include metoprolol, atorvastatin, and folic acid.", "metoprolol, atorvastatin, folic acid"),
                ("The patient's medication regimen was adjusted upon discharge.", "medication regimen adjusted")
            ],
            many=True,
        ),
        Text(
            id="discharge_diagnosis",
            description="Primary and secondary diagnoses at the time of discharge.",
            examples=[
                ("Primary diagnosis: COPD exacerbation. Secondary diagnosis: Hypertension.", "COPD exacerbation, Hypertension"),
                ("Discharge diagnosis includes respiratory failure and peptic ulcer disease.", "respiratory failure, peptic ulcer disease")
            ],
            many=True,
        ),
        Text(
            id="discharge_condition",
            description="Patient's condition at the time of discharge.",
            examples=[
                ("Patient was alert and interactive with assistance.", "alert and interactive with assistance"),
                ("Discharge condition: Confused - sometimes.", "Confused - sometimes")
            ],
            many=False,
        ),
        Text(
            id="discharge_instructions",
            description="Instructions provided to the patient at discharge.",
            examples=[
                ("Continue using nasal oxygen as needed. Follow steroid taper as instructed.", "Continue using nasal oxygen as needed., Follow steroid taper as instructed."),
                ("Patient instructed to call doctor or return to the ER if symptoms worsen.", "Call doctor or return to the ER if symptoms worsen.")
            ],
            many=True,
        ),
    ],
    many=False,
    examples=[
        (
            """Discharge Summary

                Admission Date: [**2118-6-14**]
                Discharge Date: [**2118-6-20**]
                Date of Birth: [**1937-6-12**]
                Sex: Female
                Age: 81 years

                History of Present Illness:
                An 81-year-old female with a history of emphysema, COPD, and angina presented with shortness of breath and chest pressure. She was started on a prednisone taper and oxygen therapy at home.

                Past Medical History:
                The patient has a history of emphysema, COPD, and angina.

                Physical Examination Findings:
                At the time of admission, the patient's blood pressure was 142/76 mmHg, heart rate was 100 beats per minute. Pupils were equal, round, and reactive to light. Wheezing and rhonchi were noted on auscultation. Mild edema of lower extremities and a right hand hematoma were observed.

                Laboratory Studies:
                Laboratory studies revealed a white count of 19, hematocrit of 41, platelets of 300, and Chem-7 values of 127, 3.6, 88, 29, 17, 0.6, 143.

                Major Procedures:
                During the hospital stay, the patient underwent intubation and arterial line placement.

                Medications on Admission:
                The patient was started on a prednisone taper and home oxygen therapy at admission.

                Medications on Discharge:
                Upon discharge, the patient was prescribed metoprolol, atorvastatin, and folic acid.

                Discharge Diagnosis:
                The patient was diagnosed with a COPD exacerbation and hypertension.

                Discharge Condition:
                At the time of discharge, the patient was alert and interactive with assistance.

                Discharge Instructions:
                The patient is instructed to continue using nasal oxygen as needed and to call the doctor or return to the ER if symptoms worsen.""",
            {
                "age": "81",
                "sex": "F",
                "presenting_symptoms": ["shortness of breath", "chest pressure"],
                "past_medical_history": ["emphysema", "COPD", "angina"],
                "physical_examination_findings": ["blood pressure 142/76", "heart rate 100", "equal pupils", "reactive to light"],
                "major_procedures": ["intubation", "arterial line placement"],
                "medications_on_admission": ["prednisone taper", "home oxygen"],
                "medications_on_discharge": ["metoprolol", "atorvastatin", "folic acid"],
                "discharge_diagnosis": ["COPD exacerbation", "hypertension"],
                "discharge_condition": "alert and interactive with assistance",
                "discharge_instructions": ["Continue using nasal oxygen as needed.", "Call doctor or return to ER if symptoms worsen."]
            }
        )
    ]
)

    
def extraction_chain(text_input):
    
    #Extraction Chain
    chain = create_extraction_chain(llm, 
                                    discharge_summary_schema,
                                    encoder_or_encoder_class="json", 
                                    input_formatter="triple_quotes")
    
    output = chain.run(text=text_input)["data"]
    
    return output["discharge_summary"]