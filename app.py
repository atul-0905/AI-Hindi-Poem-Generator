import warnings
import os
import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras

# Suppress warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore", message="Compiled the loaded model, but the compiled metrics have yet to be built.")

st.title("Hindi Poetry Generator")

# Load resources
with open('/Users/ganeshsapani/Documents/venv/jupyter notebook/AI hindi poetry generator/hindi-poetry-AI-master/w2i_32.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('/Users/ganeshsapani/Documents/venv/jupyter notebook/AI hindi poetry generator/hindi-poetry-AI-master/i2w_32.pkl', 'rb') as handle:
    idx_to_word = pickle.load(handle)

model = keras.models.load_model('/Users/ganeshsapani/Documents/venv/jupyter notebook/AI hindi poetry generator/hindipoet32_10.h5')

# Generate Poem
def gen_poem_random(model, length, w2i, i2w, start=' ', lines=4):
    ipseq = [w2i[s] for s in start if s in w2i]
    lsc = 0
    while lsc < lines:
        ip = np.array(pad_sequences([ipseq], maxlen=length, padding='pre'))
        w = model.predict(ip)[0]
        w = np.random.choice(np.arange(len(i2w)), p=w)
        if w == 0 or w not in i2w:
            w += 1
        if i2w[w].strip() == '<sep>':
            lsc += 1
        start += ' ' + i2w[w]
        ipseq.append(w)
    for i in start.split('<sep>'):
        st.text(i.strip())

# Input and Generate
input_text = st.text_input("Enter your initial text", "")
lines = st.number_input("Number of lines", min_value=2, max_value=12, value=4, step=1)

if st.button("Generate"):
    gen_poem_random(model, 43 - 1, tokenizer, idx_to_word, input_text, lines)
