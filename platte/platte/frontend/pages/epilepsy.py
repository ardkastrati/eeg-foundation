import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
import time
import mne
from mne.viz import plot_topomap

# Helper functions (Just for developing)!
def read_image(path, dtype=np.float32):
    f = Image.open(path)
    img = np.asarray(f, dtype)
    return img

@st.cache
def compute_sal(model_name, img, postprocessing_parameter_map, blur=0.0, hm=False):
    img = np.asarray(img, np.float32)
    return st.session_state.interface.run(model_name, img, postprocessing_parameter_map)

@st.cache
def detect_epilepsy(model_name, data):
    time.sleep(1)  # Add a 1-second delay to see the spinner
    return "0.9"
    # return st.session_state.interface.run(model_name, data)

@st.cache
def get_attention(model_name, data):
    return np.random.rand(33).tolist()
    # return st.session_state.interface.run(model_name, data)

@st.cache
def compute_test(model, original_sal, sal):
    return st.session_state.interface.test(model, original_sal, sal)

def plot_probability(probability):
    fig, ax = plt.subplots(figsize=(0.2, 0.8))
    cmap = mcolors.LinearSegmentedColormap.from_list("red_green", ["green", "red"])
    norm = plt.Normalize(0, 1)
    color = cmap(norm(probability))
    
    ax.barh([0], [1], color=color, height=0.1)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(f"{probability*100:.1f}%", fontsize=12, pad=10)
    
    return fig

def plot_topomap(attention_weights, channels):
    fig, ax = plt.subplots()
    montage = mne.channels.make_standard_montage('standard_1020')
    info = mne.create_info(ch_names=channels, sfreq=100, ch_types='eeg')
    info.set_montage(montage)
    data = np.array(attention_weights, dtype=float)
    im, _ = mne.viz.plot_topomap(data, info, names=channels, axes=ax, show=False)
    
    #pos = montage.get_positions()['ch_pos']
    #print(pos)
    #for label in channels:
    #    ch_pos = pos[label]
    #    ax.text(ch_pos[0], ch_pos[1], label, ha='center', va='center', fontsize=8, color='blue')
    
    return fig


def app():
    st.markdown("## Epileptic Seizure Detection")
    st.markdown("### Upload the brain activity of the patient.") 
    st.write("\n")

    model = "MaskedAutoEncoder"
    channels = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2", 
                "F9", "F10", "T9", "T10", "P9", "P10", "POz", "Oz", "AF3", "AF4", "CP3", "CP4", "PO3", "PO4"]

    # Code to read a single file 
    uploaded_image = st.file_uploader("Choose the file", type = ['edf', 'jpg', 'png'])

    if uploaded_image:
        original_image = Image.open(uploaded_image)

        button_col1, button_col2, button_col3 = st.columns([1, 2, 3])
        
        with button_col1:
            if st.button("Detect Epilepsy"):
                with st.spinner("Computing probability..."):
                    epilepsy_probability = float(detect_epilepsy(model, original_image))
                    st.session_state.epilepsy_probability = epilepsy_probability

        with button_col2:
            if st.button("Compute Attention"):
                attention_weights = get_attention(model, original_image)
                st.session_state.attention_weights = attention_weights

        with button_col3:
            option = st.selectbox("Choose the channel to plot:", tuple(channels))
            if st.button("Plot Brain Activity"):
                st.session_state.channel_image = "/Users/ardkastrati/Documents/YC/platte/Demo/PatientX/test.png"
                st.session_state.channel_option = option

        col0, col1, col2 = st.columns([1, 2, 3])

        if 'epilepsy_probability' in st.session_state:
            col0.markdown("### Epilepsy Probability")
            col0.pyplot(plot_probability(st.session_state.epilepsy_probability))

        if 'attention_weights' in st.session_state:
            col1.markdown("### Attention Map")
            col1.pyplot(plot_topomap(st.session_state.attention_weights, channels))

        if 'channel_image' in st.session_state:
            channel = Image.open(st.session_state.channel_image)
            col2.markdown("### Patient's Brain Activity: " + st.session_state.channel_option)
            col2.image(channel, width=500)
