import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import librosa
import os

# Assume we have the following paths. Depend on your system, it could vary
AUDIO_DIR = '/your/audio/directory'
MODEL_PATH = '/path/to/save/your/model.pt'


# The following class help transform our input into mel-spectrogram
class ToMelSpectrogram:
    def __call__(self, samples):
        return librosa.feature.melspectrogram(samples, n_mels=64, length=1024, hop_length=225)


# This class is to load audio data and apply the transformation
class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.file_list = os.listdir(self.data_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        waveform, _ = librosa.load(os.path.join(self.data_dir, self.file_list[idx]),
                                   sr=None,
                                   duration=1.0,
                                   mono=True)

        label = self.file_list[idx].split("_")[0]  # Assuming the file name is 'label_otherInfo.wav'

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, label


def train():
    # We will use the transformation to convert the audio into Mel spectrogram
    transform = Compose([ToMelSpectrogram(), ToTensor()])

    dataset = AudioDataset(AUDIO_DIR, transform=transform)
    train_set, val_set = train_test_split(dataset, test_size=0.2, stratify=dataset.targets)
    train_loader = DataLoader(dataset=train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=16, shuffle=True)

    model = CoAtNet()  # Assuming we have this class implemented following the paper or using a library
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 1100

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

        # Validation
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for inputs, labels in val_loader:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                print(f"Validation Accuracy: {correct/total}")

    torch.save(model.state_dict(), MODEL_PATH)

def main():
    train()

if __name__ == "__main__":
    main()
