import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
import yaml

def load_config(config_path="conf.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

config = load_config()

class Aggregator:
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters
        self.alpha_agg = config["aggregator"]["alpha_agg"]["selected"]
        # Load the noise parameter for Global Differential Privacy (GDP)
        self.noise_sigma = config["aggregator"].get("noise_sigma", 0.05)  # Default to 0.05 if not specified

    def cluster_clients(self, client_weights):
        num_clients = len(client_weights)
        num_clusters = min(self.num_clusters, num_clients)
        weight_matrix = np.array([np.concatenate([w.flatten() for w in client]) for client in client_weights])
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(weight_matrix)
        clusters = {}
        for i, cluster_id in enumerate(cluster_labels):
            cluster_id = int(cluster_id)
            clusters.setdefault(cluster_id, []).append(i)
        return clusters

    def aggregate(self, client_weights, clusters, client_accuracies):
        cluster_weights = {}
        for cluster_id, cluster in clusters.items():
            # Get the weights for clients in this cluster
            cluster_weights_list = [client_weights[i] for i in cluster]
            
            # Add Global Differential Privacy (GDP) noise to each client's weights
            perturbed_weights_list = []
            for weights in cluster_weights_list:
                perturbed_weights = []
                for w in weights:
                    # Add Gaussian noise with sigma from config
                    noise = np.random.normal(0, self.noise_sigma, w.shape)
                    perturbed_w = w + noise
                    perturbed_weights.append(perturbed_w)
                perturbed_weights_list.append(perturbed_weights)
            
            # Proceed with aggregation using the perturbed weights
            cluster_accs = [client_accuracies[i] for i in cluster]
            best_idx_in_cluster = cluster[np.argmax(cluster_accs)]
            best_weights = perturbed_weights_list[cluster.index(best_idx_in_cluster)]
            
            aggregated_weights = []
            for i in range(8):
                mean_noisy = np.mean([w[i] for w in perturbed_weights_list], axis=0)
                agg_layer = (self.alpha_agg * best_weights[i] + 
                            (1 - self.alpha_agg) * mean_noisy)
                aggregated_weights.append(agg_layer)
            cluster_weights[cluster_id] = aggregated_weights
        return cluster_weights