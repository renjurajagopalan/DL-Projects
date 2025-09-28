import gradio as gr
import os
import pickle

import import_ipynb
from utils import tokenize_test

from tensorflow import keras
import numpy as np

from pathlib import Path

def get_files_in_folder():
    script_path = Path(__file__).resolve()
    folder_path = script_path.parent.parent / 'models'
    print(folder_path)
    if os.path.isdir(folder_path):
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        return files
    else:
        return []
        print("I didnt find anything")


def process_inputs(sentence1, sentence2, model_name):
    print(type(sentence1))
    sentence1_pad, sentence2_pad = tokenize_test(sentence1, sentence2)

    script_path = Path(__file__).resolve()
    model_path = script_path.parent.parent / 'saved_models' / model_name
   
    model = keras.models.load_model(model_path)
    # need to load the model here

    label_path = script_path.parent / 'label_encoder_classes.pkl'
    with open(label_path, 'rb') as f:
        loaded_classes = pickle.load(f)

    test_prediction = np.argmax(model.predict([sentence1_pad, sentence2_pad]))
    predicted_class = loaded_classes[test_prediction]
    return f"Text Entailment Result: {predicted_class}"

dropdown1 = gr.Dropdown(choices=["A man inspects the uniform of a figure in some East Asian country",
                                  "An older and younger man smiling",
                                  "A black race car starts up in front of a crowd of people.",
                                  "A soccer game with multiple males playing.",
                                  "A smiling costumed woman is holding an umbrella."], label="Select Sentance 1")
dropdown2 = gr.Dropdown(choices=["The man is sleeping",
                                  "Two men are smiling and laughing at the cats playing on the floor.",
                                  "A man is driving down a lonely road.",
                                  "Some men are playing a sport.",
                                  "A happy woman in a fairy costume holds an umbrella."], label="Select Sentance2")
dropdown3 = gr.Dropdown(get_files_in_folder())

demo = gr.Interface(
fn = process_inputs,
inputs = [dropdown1, dropdown2, dropdown3],
outputs="text"
)

demo.launch()

