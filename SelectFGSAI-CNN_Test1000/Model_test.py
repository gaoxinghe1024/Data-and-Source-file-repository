import os
import json
import torch
import numpy as np
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader
from net.nn24 import SelectFGSAI_CNN

MODEL_PATH = "model_4.pth"
DATA_DIR = "Test_1000"
TEST_RESULTS_FILE = "test_results for "+MODEL_PATH+".json"
BATCH_SIZE = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class EnhancedChunkDataset(Dataset):

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.metadata = torch.load(os.path.join(data_dir, "metadata.pt"), weights_only=True)
        self.num_samples = self.metadata['total_samples']
        self.chunk_size = self.metadata['chunk_size']
        self.num_chunks = self.metadata['total_chunks']
        self.chunk_ranges = [(i * self.chunk_size, min((i + 1) * self.chunk_size, self.num_samples))
                             for i in range(self.num_chunks)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        chunk_id = idx // self.chunk_size
        local_idx = idx % self.chunk_size
        chunk_data = torch.load(
            os.path.join(self.data_dir, f"chunk_{chunk_id}.pt"),
            map_location='cpu',
            weights_only=True
        )
        tensor_data = chunk_data['data'][local_idx]
        label = int(chunk_data['labels'][local_idx][-1]) if isinstance(chunk_data['labels'][local_idx], list) else int(
            chunk_data['labels'][local_idx])

        return tensor_data, label

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found")

    model = SelectFGSAI_CNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model
def evaluate_model(model, data_loader):
    model.eval()
    all_targets = []
    all_preds = []
    running_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    i=0
    with torch.no_grad():
        for inputs, targets in data_loader:
            i = i + 1
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            print(f"\r {i}/{1000}", end="", flush=True)

    avg_loss = running_loss / len(data_loader)
    accuracy = 100 * np.mean(np.array(all_targets) == np.array(all_preds))
    report = classification_report(
        all_targets, all_preds,
        target_names=['Class 0', 'Class 1'],
        output_dict=True
    )

    cm = {
        'TP': int(((np.array(all_targets) == 1) & (np.array(all_preds) == 1)).sum()),
        'FP': int(((np.array(all_targets) == 0) & (np.array(all_preds) == 1)).sum()),
        'TN': int(((np.array(all_targets) == 0) & (np.array(all_preds) == 0)).sum()),
        'FN': int(((np.array(all_targets) == 1) & (np.array(all_preds) == 0)).sum())
    }

    return accuracy, avg_loss, report, cm


def print_results(accuracy, loss, report, cm):
    print("\n" + "=" * 60)
    print(f"{'MODEL TEST RESULTS':^60}")
    print("=" * 60)
    print(f"{'Overall Accuracy:':<20}{accuracy:.2f}%")
    print(f"{'Average Loss:':<20}{loss:.4f}\n")

    print(f"{'Class 0 Metrics:':<20}")
    print(f"{'  Precision:':<18}{report['Class 0']['precision']:.4f}")
    print(f"{'  Recall:':<18}{report['Class 0']['recall']:.4f}")
    print(f"{'  F1-score:':<18}{report['Class 0']['f1-score']:.4f}\n")

    print(f"{'Class 1 Metrics:':<20}")
    print(f"{'  Precision:':<18}{report['Class 1']['precision']:.4f}")
    print(f"{'  Recall:':<18}{report['Class 1']['recall']:.4f}")
    print(f"{'  F1-score:':<18}{report['Class 1']['f1-score']:.4f}\n")

    print(f"{'Confusion Matrix:':<20}")
    print(f"{'  TP:':<18}{cm['TP']}")
    print(f"{'  FP:':<18}{cm['FP']}")
    print(f"{'  TN:':<18}{cm['TN']}")
    print(f"{'  FN:':<18}{cm['FN']}")
    print("=" * 60)


def save_results(results, filename):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {filename}")


def main():
    print(f"Using device: {device}")

    print("\nLoading model...")
    model = load_model(MODEL_PATH)

    print("Preparing test data...")
    test_dataset = EnhancedChunkDataset(DATA_DIR)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
        num_workers=1
    )

    print("Evaluating model performance...")
    accuracy, loss, report, cm = evaluate_model(model, test_loader)

    print_results(accuracy, loss, report, cm)

    results = {
        "overall": {
            "accuracy": accuracy,
            "loss": loss
        },
        "class_0": report["Class 0"],
        "class_1": report["Class 1"],
        "confusion_matrix": cm
    }
    save_results(results, TEST_RESULTS_FILE)


if __name__ == "__main__":
    main()