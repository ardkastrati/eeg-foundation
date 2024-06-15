import os
from frontend.pages import abnormal, epilepsy
import streamlit as st
import numpy as np
from PIL import  Image

# Custom imports
from frontend.multipage import MultiPage

st.set_page_config(layout="wide")

# Create an instance of the app
app = MultiPage()

@st.cache
def load_models():
    from backend.interface import Interface
    my_interface = Interface(gpu=-1)
    return my_interface

if 'interface' not in st.session_state:
	st.session_state.interface = load_models()

#st.session_state.interface.memory_check("TEST")

# Title of the main page
display = Image.open('frontend/Images/platte.jpg')
# st.title("Data Storyteller Application")
col0, col1, col2, col3 = st.columns([5,2,7,5])
col1.image(display, width = 100)
col2.title("Platte")
# Add all your application here
app.add_page("Detect epileptic seizures", epilepsy.app)
app.add_page("Detect abnormal activities", abnormal.app)

# The main app
app.run()

st.sidebar.markdown(
"""
# Thank you!

Created by Ard Kastrati and Maxim Huber.
"""
)
