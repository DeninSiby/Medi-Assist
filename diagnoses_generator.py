from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain import PromptTemplate

   
import os

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

# Create the first prompt template
sys_prompt: PromptTemplate = PromptTemplate(
    input_variables=["Age", "Sex", "presenting_symptoms", "past_medical_history", "physical_examination_findings"],
    
    template="""Imagine you are a doctor and are given the following inputs for a patient:
                Age: {Age}/
                Sex: {Sex}/
                Presenting Symptoms: {presenting_symptoms}/
                Past Medical History: {past_medical_history}/
                Physical Examination Findings: {physical_examination_findings}/

                Generate possible diagnoses along with justifications as bullet points.""")

system_message_prompt = SystemMessagePromptTemplate(prompt=sys_prompt)

student_prompt: PromptTemplate = PromptTemplate(
    input_variables=["Age", "Sex", "presenting_symptoms", "past_medical_history", "physical_examination_findings"],
    template="""Generate Diagnoses for a patient with the following characteristics and symptoms:
    Age: {Age}/
    Sex: {Sex}/
    Presenting Symptoms: {presenting_symptoms}/
    Past Medical History: {past_medical_history}/
    Physical Examination Findings: {physical_examination_findings}"""
)

student_message_prompt = HumanMessagePromptTemplate(prompt=student_prompt)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, student_message_prompt])

# create the chat model
chat_model: ChatOpenAI = ChatOpenAI(openai_api_key=os.environ["openai_api_key"])

# Create the LLM chain
chain: LLMChain = LLMChain(llm=chat_model, prompt=chat_prompt)

def generate_diagnoses(Patient_Details):
    prediction_msg: dict = chain.run(
    Age = Patient_Details["age"], 
    Sex= Patient_Details["sex"],
    presenting_symptoms= Patient_Details["presenting_symptom"],
    past_medical_history= Patient_Details["past_medical_history"],
    physical_examination_findings= Patient_Details["physical_examination_findings"]
    )

    return prediction_msg