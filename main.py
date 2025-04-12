import tensorflow as tf
import numpy as np
import argparse
import yaml
import time
from aggregator import Aggregator
from client import Client
from dataloader import partition_data
from logging_utils import Logger

tf.config.run_functions_eagerly(True)

def load_config(config_path="conf.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def parse_args(config):
    parser = argparse.ArgumentParser(description="Federated Learning Experiment")
    parser.add_argument('--cluster', type=int, choices=config["aggregator"]["num_clusters"]["options"],
                        default=config["aggregator"]["num_clusters"]["selected"], help='Number of clusters')
    parser.add_argument('--dataset', type=str, choices=list(config["dataset"]["options"].keys()),
                        default=config["dataset"]["selected"], help='Dataset to use')
    parser.add_argument('--partitioning', type=str, choices=config["partitioning"]["options"],
                        default=config["partitioning"]["selected"], help='Data partitioning scheme')
    parser.add_argument('--alpha_agg', type=float, choices=config["aggregator"]["alpha_agg"]["options"],
                        default=config["aggregator"]["alpha_agg"]["selected"], help='Aggregation weight for best client')
    args = parser.parse_args()
    return (args.cluster, args.dataset, args.partitioning, args.alpha_agg)

def get_model_size(weights):
    """Calculate the size of model weights in bytes."""
    total_size = 0
    for w in weights:
        total_size += w.nbytes
    return total_size

def main(selected_cluster, selected_dataset, selected_partitioning, selected_alpha_agg):
    config = load_config()
    
    num_clusters = int(selected_cluster)
    dataset = str(selected_dataset)
    partitioning = str(selected_partitioning)
    config["aggregator"]["alpha_agg"]["selected"] = selected_alpha_agg
    
    if dataset not in config["dataset"]["options"]:
        raise ValueError(f"Dataset {dataset} not in {list(config['dataset']['options'].keys())}")
    num_classes = config["dataset"]["options"][dataset]["num_classes"]
    iid = partitioning == "iid"
    
    (x_train_list, y_train_list), (x_val_list, y_val_list), (x_test_list, y_test_list) = partition_data(dataset, iid)
    
    clients_data = []
    for i in range(config["client"]["num_clients"]):
        x_train = x_train_list[i] / 255.0
        y_train = tf.keras.utils.to_categorical(y_train_list[i], num_classes)
        x_val = x_val_list[i] / 255.0
        y_val = tf.keras.utils.to_categorical(y_val_list[i], num_classes)
        x_test = x_test_list[i] / 255.0
        y_test = tf.keras.utils.to_categorical(y_test_list[i], num_classes)
        clients_data.append((x_train, y_train, x_val, y_val, x_test, y_test))
    
    clients = [Client(client_id=i, dataset=clients_data[i], num_classes=num_classes) 
               for i in range(config["client"]["num_clients"])]
    
    aggregator = Aggregator(num_clusters=num_clusters)
    
    # Initialize logger for normal scenario
    logger = Logger(log_file="federated_learning_log.csv", scenario="normal")
    
    # Initialize list for round accuracies (for final summary)
    round_accuracies = []
    
    for round_num in range(config["aggregator"]["rounds"]):
        # Start timing the round
        start_time = time.time()
        
        print(f"\nRound {round_num + 1}")

        print("Initial training of clients:")
        for client in clients:
            print(f"Training client {client.client_id}...")
            client.fit()
        
        print("Evaluating clients after initial training (validation set):")
        client_val_accuracies = []
        for client in clients:
            val_acc = client.evaluate(client.x_val, client.y_val)
            client_val_accuracies.append(val_acc)
            print(f"  Client {client.client_id} validation accuracy: {val_acc:.4f}")
        
        print("Adding noise and performing aggregation:")
        client_weights = []
        # Calculate total communication cost (previously upload cost)
        total_communication_cost = 0
        for client in clients:
            noisy_weights = client.add_noise_to_parameters()
            client.set_parameters(noisy_weights)
            weights = client.get_parameters()
            total_communication_cost += get_model_size(weights)
            client_weights.append(weights)
        print(f"Total communication cost: {total_communication_cost} bytes")
        
        clusters = aggregator.cluster_clients(client_weights)
        cluster_weights = aggregator.aggregate(client_weights, clusters, client_accuracies=client_val_accuracies)
        
        print("Clusters formed:", clusters)
        print("Setting cluster-specific weights:")
        # Print the norm of the final aggregated weights for each cluster
        for cluster_id in cluster_weights:
            aggregated_weights = cluster_weights[cluster_id]
            flat_w = np.concatenate([w.flatten() for w in aggregated_weights])
            print(f"  Cluster {cluster_id} - Norm: {np.linalg.norm(flat_w):.4f}")
        
        # Set the weights for each client in the cluster
        for cluster_id, cluster in clusters.items():
            aggregated_weights = cluster_weights[cluster_id]
            for client_idx in cluster:
                client = clients[client_idx]
                new_weights = [w.numpy() if isinstance(w, tf.Tensor) else w for w in aggregated_weights]
                client.set_parameters(new_weights)
        
        print("Evaluation after update (client-specific test set):")
        client_test_accuracies = []
        for client in clients:
            acc = client.evaluate_test()
            client_test_accuracies.append(acc)
            print(f"  Client {client.client_id} accuracy: {acc:.4f}")
        
        avg_accuracy = np.mean(client_test_accuracies)
        round_accuracies.append(avg_accuracy)
        print(f"Average test accuracy for Round {round_num + 1}: {avg_accuracy:.4f}")
        
        # End timing the round
        end_time = time.time()
        round_time = end_time - start_time
        print(f"Round {round_num + 1} wall-clock time: {round_time:.2f} seconds")
        
        # Log the round's metrics
        logger.log_round(avg_accuracy, total_communication_cost, round_time)

    print("\nSimulation complete.")
    print("\nSummary of Average Test Accuracies Across Rounds:")
    for round_num, avg_acc in enumerate(round_accuracies, 1):
        print(f"Round {round_num}: {avg_acc:.4f}")

    print("\nLogs saved to federated_learning_log.csv")

if __name__ == "__main__":
    config = load_config()
    selected_cluster, selected_dataset, selected_partitioning, selected_alpha_agg = parse_args(config)
    main(selected_cluster, selected_dataset, selected_partitioning, selected_alpha_agg)