
# Image resize
IMAGE_HEIGHT = 100
IMAGE_LENGTH = 100
CHANNELS = 3

#Model params
LATENT_DIMENSION = 50
EPOCHS = 20
VALIDATION_SPLIT = 0.3
PATIENCE = 5
BATCH_SIZE = 16

# Google Cloud
BUCKET_NAME = "image-storage-stills"

# Ml flow staffs
MLFLOW_TRACKING_URI = 'https://mlflow.lewagon.ai'
MLFLOW_EXPERIMENT_AUTOENCODER = 'soundtrack_selector_autoencoder'
MLFLOW_MODEL_AUTOENCODER = 'soundtrack_selector_autoencoder'
MLFLOW_EXPERIMENT_NN = 'soundtrack_selector_nn'
MLFLOW_MODEL_NAME_NN = 'soundtrack_selector_nn'
