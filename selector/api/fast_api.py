from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from selector.ml_logic.load_model import load_model
import numpy as np


app = FastAPI()
app.state.model = load_model()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/predict")
def predict(input: np.array):
    model = app.state.model
    y_pred = model.predict(X_processed)
