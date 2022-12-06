from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from selector.ml_logic.load_model import load_model
from selector.ml_logic.proprocessor import user_image_reshape

from tensorflow.io import decode_jpeg

import numpy as np
import os

from PIL import Image


app = FastAPI()
m_enc, m_nn  = load_model()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)



app = FastAPI()

@app.post("/predict")
async def post_image(file: UploadFile = File(...)):

    #Image Send
    contents = await file.read()
    data = decode_jpeg(contents) #tensor flow format
    print(f"Started image shape: {data.shape}")

    #Charge Models
    print('Charging models ...')
    global m_enc
    global m_nn

    #PREPROCESSING
    data = user_image_reshape(data)
    print(f"Encode image shape: {data.shape}")

    #ENCODER PREDICT
    print(f'Making magic...')
    image_encode = m_enc.predict(data)

    #NN PREDICT
    print(f'Serching for your Match...')
    indices = m_nn.kneighbors(image_encode)[1]


    print(f'Process Ready ✅✅✅')
    return {"idx":[int(x) for x in list(indices[0])]}#list(indices)}


#@app.post("/predict")
#async def predict_image(image: Request):
#    global m_enc
#    global m_nn
#
#    data = await image.json()
#    data = np.array(eval(data))
#    #PREPROCESSING
#    data = user_image_reshape(data)
#    print(data.shape)
#    #ENCODER PREDICT
#    image_encode = m_enc.predict(data)
#    #NN PREDICT
#    indices = m_nn.kneighbors(image_encode)[1]
#    return {"idx":list(indices)}
#
