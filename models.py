import tensorflow as tf
from tensorflow.keras import layers, models
import yaml

def load_config(config_path="conf.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

config = load_config()

def create_cnn_model(num_classes):
    return models.Sequential([
        layers.Conv2D(20, kernel_size=(5, 5), activation='relu', input_shape=config["model"]["input_shape"]),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(50, kernel_size=(5, 5), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(500, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])