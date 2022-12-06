from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from selector.ml_logic.load_model import load_model
from selector.ml_logic.proprocessor import user_image_reshape

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


@app.post("/predict")
async def predict_image(image: Request):
    global m_enc
    global m_nn

    data = await image.json()
    data = np.array(eval(data))
    #PREPROCESSING
    data = user_image_reshape(data)
    print(data.shape)
    #ENCODER PREDICT
    image_encode = m_enc.predict(data)
    #NN PREDICT
    indices = m_nn.kneighbors(image_encode)[1]
    return {"idx":list(indices)}
