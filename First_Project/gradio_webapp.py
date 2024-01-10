import gradio as gr
import requests
import random
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import argparse
import torch
import torchvision
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer

#Getting the model
model= models.mobilenet_v3_small(weights='MobileNet_V3_Small_Weights.IMAGENET1K_V1')
model.classifier= nn.Flatten()
model.eval()


#Our dataframe
df_image = pd.read_pickle('./movies.pkl')
df_text = pd.read_pickle ('./metadata.pkl')



def process_image(image):
    
    # Spécifiez les moyennes et les écart-types
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Définissez les transformations
    normalize = transforms.Normalize(mean, std)


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    # Appliquez les transformations à l'image
    pil_image = Image.fromarray(image)
    tensor_image = transform(pil_image).unsqueeze(0)

    vector = model(tensor_image).tolist()[0]
      
    response = requests.post('http://annoy-db:5000/reco', json={'vector': vector})
    # Pour tester sans docker 
    #response = requests.post('http://127.0.0.1:5000/reco', json={'vector': vector})
    if response.status_code == 200:
        indices = response.json()
       
        # Retrieve paths for the indices
        paths = df_image.loc[indices, 'path']
        print(f'PATHS = {paths}')

        # Plot the images
        fig, axs = plt.subplots(1, len(paths), figsize=(5 * len(paths), 5))
        for i, path in enumerate(paths):
            img = Image.open(path)
            axs[i].imshow(img)
            axs[i].axis('off')
        return fig
    else:
        return "Error in API request"

def process_text(text: str):
       
    response = requests.post('http://annoy-db:5000/api/search', json={'text': text})
    # Pour tester sans docker 
    #response = requests.post('http://127.0.0.1:5000/api/search', json={'text': text})
    
    if response.status_code == 200:
        indices = response.json()
        print(f'INDICES = {indices}')
        
        # Retrieve paths for the indices
        
        titles = df_text.loc[indices, 'title'].values.tolist()
   
        print(f'TITLES = {titles}')
        
        fig, ax = plt.subplots(figsize=(10, 5))

        # Affichez tous les titres horizontalement
        ax.axis('off')  # Désactivez les axes
        ax.text(0, 0.5, '\n'.join(titles), fontsize=40, ha='left', va='center')  # Affichez les titres

        return fig

    else:
        return "Error in API request"


def process_text_glove(text: str):
       
    response = requests.post('http://annoy-db:5000/api/glove', json={'text': text})
    # Pour tester sans docker
    #response = requests.post('http://127.0.0.1:5000/api/glove', json={'text': text})
    
    if response.status_code == 200:
        indices = response.json()
        print(f'INDICES = {indices}')
        
        # Retrieve paths for the indices
        
        titles = df_text.loc[indices, 'title'].values.tolist()
   
        print(f'TITLES = {titles}')
        
        fig, ax = plt.subplots(figsize=(10, 5))

        # Affichez tous les titres horizontalement
        ax.axis('off')  # Désactivez les axes
        ax.text(0, 0.5, '\n'.join(titles), fontsize=40, ha='left', va='center')  # Affichez les titres

        return fig

    else:
        return "Error in API request"
    

iface1= gr.Interface(fn=process_image, inputs="image", outputs="plot")
iface2= gr.Interface(fn=process_text, inputs="text", outputs="plot")
iface3= gr.Interface(fn=process_text_glove, inputs="text", outputs="plot")
gr.TabbedInterface([iface1,iface2, iface3],["Image","Text","Text"]).launch(server_name="0.0.0.0") # the server will be accessible externally under this address


