# AI Hindi Poem Generator

This is an AI-based Hindi Poem Generator built using **TensorFlow** and **Keras**. It generates Hindi poems based on user input (a theme or keyword). The model is trained from scratch using a custom Hindi poetry dataset and word embeddings.


## 🔧 Technologies Used
- Python
- TensorFlow & Keras
- Word Embeddings (FastText)
- NumPy / Pickle


## 📁 Project Files

- `app.py` – Main script to run the generator
- `AI Hindi Poem Generator.ipynb` – Jupyter Notebook (model training and testing)
- `cc.hi.300.vec` – Pre-trained FastText Hindi word vectors
- `embedding_matrix_32.pkl` – Word embedding matrix used for training
- `hindipoet32_5.h5` / `10.h5` / `12.h5` – Trained model files
- `i2w_32.pkl` / `w2i_32.pkl` – Word-to-index and index-to-word mappings
- `requirements.txt` – Python dependencies


## 🚀 How to Run

1. Install dependencies:
   pip install -r requirements.txt

   
2. Run the script:
   python app.py
   
3. Enter a Hindi word or theme when prompted. The model will generate a poem based on your input.


📌 Notes
The model is trained from scratch (no use of pre-trained text generation models like GPT).
FastText Hindi vectors (cc.hi.300.vec) were used for embeddings.
Easily extendable to a web interface using Streamlit or Flask.

