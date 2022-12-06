import os
import mlflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input
from dotenv import load_dotenv, find_dotenv

env_path = find_dotenv()
load_dotenv(env_path)


def load_model():
    """
    load the latest saved model, return None if no model found
    """
    stage = "Production"

    # load encoder model from mlflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    model_uri_autoenc = 'models:/soundtrack_selector_autoencoder/Production'
    model_uri_nn = 'models:/soundtrack_selector_nn/Production'

    # load near neighboard model from mlflow

    print(f"- uri: {model_uri_autoenc}")
    print(f"- uri: {model_uri_nn}")

    try:
        model_autoenc = mlflow.keras.load_model(model_uri=model_uri_autoenc)
        model_nn = mlflow.sklearn.load_model(model_uri=model_uri_nn)
        #AUTOENCODER
        model_enc = Sequential([
            Input((100, 100, 3), name="REAL_INPUT"),
            model_autoenc.layers[0],
            model_autoenc.layers[1]
        ])
        print("\n✅ Encouder and NN models loaded from mlflow")
    except:
        print(f"\n❌ no model in stage {stage} on mlflow")
        raise Exception("No models")

    return model_enc, model_nn
