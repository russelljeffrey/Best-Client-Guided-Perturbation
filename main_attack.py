import tensorflow as tf
import numpy as np
import argparse
import yaml
import time
from aggregator import Aggregator
from client import Client, MaliciousClient
from dataloader import partition_data
from logging_utils import Logger

tf.config.run_functions_eagerly(True)

def load_config(config_path="conf.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def parse_args(config):
    parser = argparse.ArgumentParser(description="Federated Learning Attack Experiment")
    parser.add_argument('--cluster', type=int, choices=config["aggregator"]["num_clusters"]["options"],
                        default=config["aggregator"]["num_clusters"]["selected"], help='Number of clusters')
    parser.add_argument('--dataset', type=str, choices=list(config["dataset"]["options"].keys()),
                        default=config["dataset"]["selected"], help='Dataset to use')
    parser.add_argument('--partitioning', type=str, choices=config["partitioning"]["options"],
                        default=config["partitioning"]["selected"], help='Data partitioning scheme')
    parser.add_argument('--malicious_clients', type=int, choices=config["attack"]["malicious_clients"]["options"],
                        default=config["attack"]["malicious_clients"]["selected"], help='Number of malicious clients')
    parser.add_argument('--attack', type=str, choices=["backdoor", "feature_manipulation"],
                        help='Type of attack to perform')
    parser.add_argument('--poisoning_percentage', type=float, choices=config["attack"]["poisoning_percentage"]["options"],
                        default=config["attack"]["poisoning_percentage"]["selected"], help='Percentage of data to poison')
    args = parser.parse_args()
    return (args.cluster, args.dataset, args.partitioning, args.malicious_clients, args.attack, args.poisoning_percentage)

def get_model_size(weights):
    """Calculate the size of model weights in bytes."""
    total_size = 0
    for w in weights:
        total_size += w.nbytes
    return total_size

def main(selected_cluster, selected_dataset, selected_partitioning, num_malicious_clients, attack_type, poisoning_percentage):
    config = load_config()
    
    num_clusters = int(selected_cluster)
    dataset = str(selected_dataset)
    partitioning = str(selected_partitioning)
    num_malicious_clients = int(num_malicious_clients)
    attack_type = str(attack_type)
    poisoning_percentage = float(poisoning_percentage)
    
    if dataset not in config["dataset"]["options"]:
        raise ValueError(f"Dataset {dataset} not in {list(config['dataset']['options'].keys())}")
    num_classes = config["dataset"]["options"][dataset]["num_classes"]
    iid = partitioning == "iid"
    
    # random seed for reproducibility
    # random_seed = config["attack"]["random_seed"]
    # np.random.seed(random_seed)
    # tf.random.set_seed(random_seed)

    # Randomly select target classes for the attacks
    backdoor_target_class = np.random.randint(0, num_classes)
    feature_manipulation_target_class = np.random.randint(0, num_classes)
    while feature_manipulation_target_class == backdoor_target_class:
        feature_manipulation_target_class = np.random.randint(0, num_classes)
    
    # Select target class based on attack type
    target_class = backdoor_target_class if attack_type == "backdoor" else feature_manipulation_target_class
    print(f"Attack type: {attack_type}, Target class: {target_class}, Poisoning percentage: {poisoning_percentage}")

    # Randomly select malicious clients
    all_client_ids = list(range(config["client"]["num_clients"]))
    malicious_client_ids = np.random.choice(all_client_ids, num_malicious_clients, replace=False)
    print(f"Malicious clients: {malicious_client_ids}")

    # Load and partition data
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
    
    # Create clients (normal or malicious)
    clients = []
    for i in range(config["client"]["num_clients"]):
        if i in malicious_client_ids:
            client = MaliciousClient(client_id=i, dataset=clients_data[i], num_classes=num_classes,
                                     target_class=target_class, attack_type=attack_type,
                                     poisoning_percentage=poisoning_percentage)
        else:
            client = Client(client_id=i, dataset=clients_data[i], num_classes=num_classes)
        clients.append(client)
    
    aggregator = Aggregator(num_clusters=num_clusters)
    
    # Initialize logger for attack scenarios
    logger = Logger(log_file="attack_log.csv", scenario="attack")
    
    # Initialize lists for round metrics (for final summary)
    round_test_errors = []
    round_test_accuracies = []
    
    for round_num in range(config["aggregator"]["rounds"]):
        # Start timing the round
        start_time = time.time()
        
        print(f"\nRound {round_num + 1}")

        print("Initial training of clients:")
        for client in clients:
            print(f"Training client {client.client_id}...")
            # Apply attack for malicious clients before training
            if isinstance(client, MaliciousClient):
                client.apply_attack()
            client.fit()
        
        print("Evaluating clients after initial training (validation set):")
        client_val_accuracies = []
        for client in clients:
            val_acc = client.evaluate(client.x_val, client.y_val)
            client_val_accuracies.append(val_acc)
            print(f"  Client {client.client_id} validation accuracy: {val_acc:.4f}")
        
        print("Adding noise and performing aggregation:")
        client_weights = []
        for client in clients:
            noisy_weights = client.add_noise_to_parameters()
            client.set_parameters(noisy_weights)
            weights = client.get_parameters()
            client_weights.append(weights)
        
        clusters = aggregator.cluster_clients(client_weights)
        cluster_weights = aggregator.aggregate(client_weights, clusters, client_accuracies=client_val_accuracies)
        
        print("Clusters formed:", clusters)
        print("Setting cluster-specific weights:")
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
        client_test_errors = []
        for client in clients:
            acc = client.evaluate_test()
            error = 1.0 - acc
            client_test_accuracies.append(acc)
            client_test_errors.append(error)
            print(f"  Client {client.client_id} test accuracy: {acc:.4f}, test error: {error:.4f}")
        
        avg_test_accuracy = np.mean(client_test_accuracies)
        avg_test_error = np.mean(client_test_errors)
        round_test_accuracies.append(avg_test_accuracy)
        round_test_errors.append(avg_test_error)
        print(f"Average test accuracy for Round {round_num + 1}: {avg_test_accuracy:.4f}")
        print(f"Average test error for Round {round_num + 1}: {avg_test_error:.4f}")
        
        # End timing the round
        end_time = time.time()
        round_time = end_time - start_time
        print(f"Round {round_num + 1} wall-clock time: {round_time:.2f} seconds")
        
        # Log the round's metrics
        logger.log_attack_round(avg_test_error, avg_test_accuracy, round_time)

    print("\nSimulation complete.")
    print("\nSummary of Average Test Errors and Accuracies Across Rounds:")
    for round_num, (avg_err, avg_acc) in enumerate(zip(round_test_errors, round_test_accuracies), 1):
        print(f"Round {round_num}: Test Error = {avg_err:.4f}, Test Accuracy = {avg_acc:.4f}")

    print("\nLogs saved to attack_log.csv")

if __name__ == "__main__":
    config = load_config()
    selected_cluster, selected_dataset, selected_partitioning, num_malicious_clients, attack_type, poisoning_percentage = parse_args(config)
    main(selected_cluster, selected_dataset, selected_partitioning, num_malicious_clients, attack_type, poisoning_percentage)