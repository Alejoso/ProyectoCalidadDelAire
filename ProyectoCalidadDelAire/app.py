from openai import OpenAI
from PIL import Image
import streamlit as st
import torch
from diffusers import AutoPipelineForText2Image
import torch
from pandasai import SmartDataframe
import pandas as pd

from pandasai.llm import OpenAI
from pandasai.responses.response_parser import ResponseParser
import os

api_token = "sk-proj-yHAvHtH6D92MGxxXdYV5T3BlbkFJdUxvRyzUcvbc2BqDFhQ5" #En caso de tener una con billing, cambiar la key

st.set_page_config(page_title="Air Pollution", page_icon=":bar_chart:", layout="wide")

st.title("Air Pollution In Medellin")
st.subheader("On this website, you can navigate trougth the latest information about air pollution in Medellin or generate images according to the topic")

#Imagen centrada 
col1, col2, col3 = st.columns(3)

with col1:
    st.write(' ')

with col2:
    st.image("./img/medellin.png" , width= 500 )

with col3:
    st.write(' ')

##Creacion de las tablas

class StreamlitResponse(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        st.dataframe(result["value"])
        return

    def format_plot(self, result):
        st.image(result["value"])
        return

    def format_other(self, result):
        st.write(result["value"])
        return

st.subheader("What is pm 2.5?")    

st.write("Particle pollution from fine particulates (PM2.5) is a concern when levels in air are unhealthy. Breathing in unhealthy levels of PM2.5 can increase the risk of health problems like heart disease, asthma, and low birth weight. Unhealthy levels can also reduce visibility and cause the air to appear hazy.")

st.title("Pm 2.5 data")

df = pd.read_csv("./data/Datos.csv")
with st.expander("🔎Here you can see the latest data about pm 2.5"):
    st.write(df.tail(10))

query = st.text_area("🗣️ Chat with Dataframe")
container = st.container()

if query:
    llm = OpenAI(api_token)
    query_engine = SmartDataframe(
        df,
        config={
            "llm": llm,
            "response_parser": StreamlitResponse,
        },
    )

    answer = query_engine.chat(query)
    st.write(answer)

st.title("Create an image!")

img_description = st.text_input("Write the prompt for the image you want to create")


modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda"
auth_token = "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"  


if st.button("Generate image"):
    pipeline = AutoPipelineForText2Image.from_pretrained(
	"runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")
    image = pipeline(img_description).images[0]
    st.image(image)