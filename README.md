# Best-Client Guided Perturbation

This repository contains code for the *Modeling Decisions for Artificial Intelligence (MDAI) 2025* conference submission. It implements federated learning simulations with support for normal client behavior and malicious attacks, including backdoor, label flipping, and feature manipulation.

## Usage

- **`main.py`**: Simulates standard federated learning clients.
- **`main_attack.py`**: Simulates malicious clients with backdoor, label flipping, or feature manipulation attacks.
- **Configuration**: Adjust parameters like the number of malicious clients in `conf.yaml`.

### Running Normal Federated Learning

Example command:
```bash
python main.py --cluster 5 --dataset femnist --partitioning noniid --alpha_agg 0.5
```

**Parameters**:
- `--cluster`: Number of clusters (3 or 5, defined in `conf.yaml`).
- `--dataset`: Dataset name (`mnist`, `fashion_mnist`, or `femnist`).
- `--partitioning`: Data partitioning type (`iid` or `noniid`).
- `--alpha_agg`: Best-client contribution weight (0.1, 0.3, or 0.5).

### Running Attack Simulations

Example command:
```bash
python main_attack.py --cluster 5 --dataset femnist --partitioning noniid --malicious_clients 1 --attack backdoor
```

**Parameters**:
- `--cluster`: Number of clusters (3 or 5, defined in `conf.yaml`).
- `--dataset`: Dataset name (`mnist`, `fashion_mnist`, or `femnist`).
- `--partitioning`: Data partitioning type (`iid` or `noniid`).
- `--malicious_clients`: Number of malicious clients (1 or 2 out of 10).
- `--attack`: Attack type (`backdoor` or `feature_manipulation`).
- `--alpha_agg`: Fixed at 0.3 for attack simulations.
- **Note**: Backdoor attacks include label flipping.
