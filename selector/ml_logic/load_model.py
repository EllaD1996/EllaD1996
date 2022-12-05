import os
import mlflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input


def load_model():
    """
    load the latest saved model, return None if no model found
    """
    stage = "Production"

    # load encoder model from mlflow
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

    mlflow_autoencoder = os.environ.get("MLFLOW_MODEL_NAME_AUTOENCODER")
    mlflow_nn = os.environ.get("MLFLOW_MODEL_NAME_NN")

    # ENCODER
    IMAGE_HEIGHT = os.environ.get("IMAGE_HEIGHT")
    IMAGE_LENGTH = os.environ.get("IMAGE_LENGTH")
    CHANNELS = os.environ.get("CHANNELS")


    model_uri_autoenc = f"models:/{mlflow_autoencoder}/{stage}"
    model_uri_nn = f"models:/{mlflow_nn}/{stage}"

    # load near neighboard model from mlflow

    print(f"- uri: {model_uri_autoenc}")
    print(f"- uri: {model_uri_nn}")

    try:
        model_autoenc = mlflow.keras.load_model(model_uri=model_uri_autoenc)
        model_nn = mlflow.sklearn.load_model(model_uri=model_uri_nn)
        #AUTOENCODER
        model_enc = Sequential([
            Input((IMAGE_HEIGHT, IMAGE_LENGTH, CHANNELS)),
            model_autoenc.layers[0],
            model_autoenc.layers[1]
        ])
        print("\n✅ Encouder and NN models loaded from mlflow")
    except:
        print(f"\n❌ no model in stage {stage} on mlflow")
        return None

    return model_enc, model_nn
