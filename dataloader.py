import numpy as np
import yaml
from torchvision import datasets, transforms

def load_config(config_path="conf.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

config = load_config()

def dirichlet_split(n_clients, n_classes, alpha, n_samples_per_client, labels):
    """Distribute samples to clients using Dirichlet distribution, ensuring exact sample counts."""
    # Step 1: Compute class proportions using Dirichlet distribution
    min_size = 0
    while min_size < 1:
        proportions = np.random.dirichlet(np.repeat(alpha, n_clients), n_classes)
        proportions = proportions / proportions.sum(axis=0, keepdims=True)
        class_counts = (proportions * n_samples_per_client).astype(int)
        min_size = np.min(class_counts.sum(axis=1))
    
    # Step 2: Adjust class counts to ensure each client gets exactly n_samples_per_client
    class_counts = (proportions * n_samples_per_client).astype(int)
    for i in range(n_clients):
        total = class_counts[:, i].sum()
        if total < n_samples_per_client:
            deficit = n_samples_per_client - total
            idx = np.argmax(class_counts[:, i])
            class_counts[idx, i] += deficit
        elif total > n_samples_per_client:
            excess = total - n_samples_per_client
            idx = np.argmax(class_counts[:, i])
            class_counts[idx, i] -= excess
    
    # Step 3: Sample indices for each client based on class counts
    client_indices = [[] for _ in range(n_clients)]
    for class_id in range(n_classes):
        class_indices = np.where(labels == class_id)[0]
        if len(class_indices) == 0:
            continue
        for client_id in range(n_clients):
            num_samples = class_counts[class_id, client_id]
            if num_samples > 0:
                # Sample with replacement if needed
                selected_indices = np.random.choice(class_indices, num_samples, replace=True)
                client_indices[client_id].extend(selected_indices)
    
    return client_indices

def load_femnist():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.EMNIST(root='./data', split='byclass', train=True, download=True, transform=transform)
    test_dataset = datasets.EMNIST(root='./data', split='byclass', train=False, download=True, transform=transform)

    x_train = train_dataset.data.numpy()
    y_train = train_dataset.targets.numpy()
    x_test = test_dataset.data.numpy()
    y_test = test_dataset.targets.numpy()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')

    return (x_train, y_train), (x_test, y_test)

def partition_data(dataset_name, iid=True):
    if dataset_name == "mnist":
        import tensorflow as tf
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')
        x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')
    elif dataset_name == "fashion_mnist":
        import tensorflow as tf
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')
        x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')
    elif dataset_name == "femnist":
        (x_train, y_train), (x_test, y_test) = load_femnist()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    num_clients = config["client"]["num_clients"]
    requested_samples_per_client = config["client"]["samples_per_client"]
    train_size = config["client"]["train_size"]
    val_size = config["client"]["val_size"]
    test_size = config["client"]["test_size"]
    n_classes = config["dataset"]["options"][dataset_name]["num_classes"]

    # Verify total matches samples_per_client in config
    total_requested = train_size + val_size + test_size
    if total_requested != requested_samples_per_client:
        raise ValueError(f"Total requested samples ({total_requested}) does not match samples_per_client ({requested_samples_per_client})")

    # Compute total required samples
    train_val_samples_per_client = train_size + val_size
    total_train_val_samples = num_clients * train_val_samples_per_client
    total_test_samples = num_clients * test_size

    # Oversample training data if necessary
    original_train_size = len(x_train)
    if original_train_size < total_train_val_samples:
        print(f"Oversampling training data for {dataset_name}: original size={original_train_size}, required={total_train_val_samples}")
        indices = np.random.choice(original_train_size, total_train_val_samples, replace=True)
        x_train = x_train[indices]
        y_train = y_train[indices]
    else:
        indices = np.random.permutation(original_train_size)
        x_train = x_train[indices]
        y_train = y_train[indices]

    # Oversample test data if necessary
    original_test_size = len(x_test)
    if original_test_size < total_test_samples:
        print(f"Oversampling test data for {dataset_name}: original size={original_test_size}, required={total_test_samples}")
        indices = np.random.choice(original_test_size, total_test_samples, replace=True)
        x_test = x_test[indices]
        y_test = y_test[indices]
    else:
        indices = np.random.permutation(original_test_size)
        x_test = x_test[indices]
        y_test = y_test[indices]

    x_train_list, y_train_list = [], []
    x_val_list, y_val_list = [], []
    x_test_list, y_test_list = [], []

    # Partition training data (for train and val)
    if iid:
        for i in range(num_clients):
            start_idx = i * train_val_samples_per_client
            end_idx = (i + 1) * train_val_samples_per_client
            client_data = x_train[start_idx:end_idx]
            client_labels = y_train[start_idx:end_idx]

            # Since we oversampled, we should always have enough data
            assert len(client_data) == train_val_samples_per_client, f"Client {i} train/val data mismatch: {len(client_data)} vs {train_val_samples_per_client}"

            indices = np.random.permutation(len(client_data))
            client_data, client_labels = client_data[indices], client_labels[indices]
            x_train_client = client_data[:train_size]
            y_train_client = client_labels[:train_size]
            x_val_client = client_data[train_size:train_size + val_size]
            y_val_client = client_labels[train_size:train_size + val_size]

            x_train_list.append(x_train_client)
            y_train_list.append(y_train_client)
            x_val_list.append(x_val_client)
            y_val_list.append(y_val_client)
    else:
        alpha_dir = config["dirichlet"]["alpha"]
        client_indices = dirichlet_split(num_clients, n_classes, alpha_dir, train_val_samples_per_client, y_train)
        for i in range(num_clients):
            indices = client_indices[i]
            if len(indices) == 0:
                print(f"Client {i}: No train/val data assigned (non-IID).")
                client_data = np.array([]).reshape(0, 28, 28, 1)
                client_labels = np.array([])
            else:
                client_data = x_train[indices]
                client_labels = y_train[indices]

            # Since dirichlet_split ensures exact counts, this should match
            assert len(client_data) == train_val_samples_per_client, f"Client {i} train/val data mismatch: {len(client_data)} vs {train_val_samples_per_client}"

            indices = np.random.permutation(len(client_data))
            client_data, client_labels = client_data[indices], client_labels[indices]
            x_train_client = client_data[:train_size]
            y_train_client = client_labels[:train_size]
            x_val_client = client_data[train_size:train_size + val_size]
            y_val_client = client_labels[train_size:train_size + val_size]

            x_train_list.append(x_train_client)
            y_train_list.append(y_train_client)
            x_val_list.append(x_val_client)
            y_val_list.append(y_val_client)

    # Partition test data separately
    if iid:
        for i in range(num_clients):
            start_idx = i * test_size
            end_idx = (i + 1) * test_size
            x_test_client = x_test[start_idx:end_idx]
            y_test_client = y_test[start_idx:end_idx]

            assert len(x_test_client) == test_size, f"Client {i} test data mismatch: {len(x_test_client)} vs {test_size}"

            x_test_list.append(x_test_client)
            y_test_list.append(y_test_client)
    else:
        alpha_dir = config["dirichlet"]["alpha"]
        client_indices = dirichlet_split(num_clients, n_classes, alpha_dir, test_size, y_test)
        for i in range(num_clients):
            indices = client_indices[i]
            if len(indices) == 0:
                print(f"Client {i}: No test data assigned (non-IID).")
                x_test_client = np.array([]).reshape(0, 28, 28, 1)
                y_test_client = np.array([])
            else:
                x_test_client = x_test[indices]
                y_test_client = y_test[indices]

            assert len(x_test_client) == test_size, f"Client {i} test data mismatch: {len(x_test_client)} vs {test_size}"

            x_test_list.append(x_test_client)
            y_test_list.append(y_test_client)

    # Print debug information
    for i in range(num_clients):
        print(f"Client {i}: train samples = {len(x_train_list[i])}, val samples = {len(x_val_list[i])}, test samples = {len(x_test_list[i])}")

    return (x_train_list, y_train_list), (x_val_list, y_val_list), (x_test_list, y_test_list)