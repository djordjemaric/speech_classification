from torchaudio.datasets import SPEECHCOMMANDS
import os

# Define a custom dataset class for easier handling
class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__(root="./data", download=True)
        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as file:
                return [os.path.join(self._path, line.strip()) for line in file]
        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

# Load datasets
train_set = SubsetSC("training")
val_set = SubsetSC("validation")
test_set = SubsetSC("testing")

# Print dataset sizes
print(f"Training set size: {len(train_set)}")
print(f"Validation set size: {len(val_set)}")
print(f"Testing set size: {len(test_set)}")

# Example: Accessing a data sample
waveform, sample_rate, label, *_ = train_set[0]
print(f"Waveform shape: {waveform.shape}")
print(f"Sample rate: {sample_rate}")
print(f"Label: {label}")





