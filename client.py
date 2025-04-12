import tensorflow as tf
import numpy as np
from models import create_cnn_model
import yaml

def load_config(config_path="conf.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

config = load_config()

class Client:
    def __init__(self, client_id, dataset, num_classes):
        self.client_id = client_id
        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test = dataset
        
        self.model = create_cnn_model(num_classes)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config["client"]["learning_rate"]),
                           loss='categorical_crossentropy', metrics=['accuracy'])

    def get_parameters(self):
        return self.model.get_weights()

    def set_parameters(self, parameters):
        self.model.set_weights(parameters)

    def fit(self):
        if len(self.x_train) == 0:
            print(f"Client {self.client_id}: No training data available, skipping fit.")
            return
        steps_per_epoch = max(1, (len(self.x_train) + config["client"]["batch_size"] - 1) // config["client"]["batch_size"])
        self.model.fit(self.x_train, self.y_train, 
                       epochs=config["client"]["local_epochs"],
                       batch_size=config["client"]["batch_size"],
                       steps_per_epoch=steps_per_epoch,
                       verbose=0)

    def evaluate(self, x, y):
        if len(x) == 0:
            print(f"Client {self.client_id}: No evaluation data available, returning 0 accuracy.")
            return 0.0
        _, accuracy = self.model.evaluate(x, y, verbose=0)
        return accuracy

    def evaluate_test(self):
        return self.evaluate(self.x_test, self.y_test)

    def add_noise_to_parameters(self):
        weights = self.model.get_weights()
        sigma = config["client"]["noise"]["sigma"]
        noisy_weights = []
        for w in weights:
            noise = np.random.normal(0, sigma, w.shape)
            noisy_weights.append(w + noise)
        return noisy_weights

class MaliciousClient(Client):
    def __init__(self, client_id, dataset, num_classes, target_class, attack_type, poisoning_percentage):
        super().__init__(client_id, dataset, num_classes)
        self.target_class = target_class
        self.attack_type = attack_type
        self.poisoning_percentage = poisoning_percentage  # Store the poisoning percentage

    def poison_data_backdoor(self):
        """Poison training data for Backdoor Attack: add 5x5 white square and flip labels."""
        num_samples = len(self.x_train)
        num_poisoned = int(num_samples * self.poisoning_percentage)

        # Randomly select indices to poison
        indices = np.random.choice(num_samples, num_poisoned, replace=False)

        # Copy the training data to avoid modifying the original
        x_poisoned = self.x_train.copy()
        y_poisoned = self.y_train.copy()

        # Add 5x5 white square trigger in the bottom-right corner
        trigger_size = config["attack"]["backdoor_trigger"]["size"]
        trigger_value = config["attack"]["backdoor_trigger"]["value"]
        start_row = 28 - trigger_size  # 28 - 5 = 23
        start_col = 28 - trigger_size  # 28 - 5 = 23
        for idx in indices:
            x_poisoned[idx, start_row:start_row + trigger_size, start_col:start_col + trigger_size, 0] = trigger_value
            # Flip the label to the target class
            y_poisoned[idx] = np.zeros_like(y_poisoned[idx])
            y_poisoned[idx, self.target_class] = 1

        self.x_train = x_poisoned
        self.y_train = y_poisoned

    def poison_data_feature_manipulation(self):
        """Poison training data for Feature Manipulation Attack: add Gaussian noise to target class images."""
        num_samples = len(self.x_train)
        num_poisoned = int(num_samples * self.poisoning_percentage)

        # Randomly select indices to poison
        indices = np.random.choice(num_samples, num_poisoned, replace=False)

        # Copy the training data to avoid modifying the original
        x_poisoned = self.x_train.copy()
        y_poisoned = self.y_train.copy()

        # Add Gaussian noise to images of the target class
        noise_budget = config["attack"]["noise_budget"]
        for idx in indices:
            # Check if the sample belongs to the target class
            if np.argmax(y_poisoned[idx]) == self.target_class:
                noise = np.random.normal(0, noise_budget, x_poisoned[idx].shape)
                x_poisoned[idx] = np.clip(x_poisoned[idx] + noise, 0.0, 1.0)

        self.x_train = x_poisoned
        self.y_train = y_poisoned

    def apply_attack(self):
        """Apply the specified attack to the training data."""
        if self.attack_type == "backdoor":
            self.poison_data_backdoor()
        elif self.attack_type == "feature_manipulation":
            self.poison_data_feature_manipulation()
        else:
            raise ValueError(f"Unsupported attack type: {self.attack_type}")