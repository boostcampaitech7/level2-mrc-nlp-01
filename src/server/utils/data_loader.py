import streamlit as st
from datasets import load_from_disk

@st.cache_resource
def load_dataset():
    data = load_from_disk("./data/train_dataset")
    return data