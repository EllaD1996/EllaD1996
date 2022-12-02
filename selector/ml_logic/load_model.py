import os
import mlflow


from colorama import Fore, Style

def load_model() -> Model:
    """
    load the latest saved model, return None if no model found
    """
    stage = "Production"

    print(Fore.BLUE + f"\nLoad model {stage} stage from mlflow..." + Style.RESET_ALL)

    # load encoder model from mlflow
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

    mlflow_encouder_model_name = os.environ.get("MLFLOW_MODEL_NAME")
    mlflow_encouder_model_name = os.environ.get("MLFLOW_MODEL_NAME")
    model_uri = f"models:/{mlflow_model_name}/{stage}"

    # load near neighboard model from mlflow

    print(f"- uri: {model_uri}")

    try:
        model = mlflow.keras.load_model(model_uri=model_uri)
        print("\n✅ Encouder model loaded from mlflow")
    except:
        print(f"\n❌ no model in stage {stage} on mlflow")
        return None

    return model
