from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import librosa
import soundfile as sf
import os
import numpy as np
import scipy.ndimage as ndimage
from DatasetLoading import resample_spectrogram_db, resample_spectrogram_phase, make_lengths_same


# Dataset class
class UNetDataset(Dataset):
    def __init__(self, input_dir, output_dir, transform=None, target_transform=None, image_size=(512, 512)):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.transform = transform
        self.target_transform = target_transform
        self.image_size = image_size

        # List all input files
        self.input_files = sorted(os.listdir(input_dir))

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        # Load input image
        input_file = self.input_files[idx]
        input_path = os.path.join(self.input_dir, input_file)
        input_image = Image.open(input_path).convert('L')  # Convert to grayscale

        # Load corresponding output folder
        track_name = input_file.split('_mix')[0]
        output_folder = os.path.join(self.output_dir, track_name)
        output_files = sorted(os.listdir(output_folder))

        # Load and stack output images
        output_images = []
        for output_file in output_files:
            output_path = os.path.join(output_folder, output_file)
            output_image = Image.open(output_path).convert('L')  # Convert to grayscale
            output_images.append(output_image)

        # Resize input and output images
        if self.image_size:
            input_image = input_image.resize(self.image_size)
            output_images = [img.resize(self.image_size) for img in output_images]

        # Apply transformations
        if self.transform:
            input_image = self.transform(input_image)
        else:
            input_image = transforms.ToTensor()(input_image)  # Default transform to tensor

        if self.target_transform:
            output_images = [self.target_transform(img) for img in output_images]
        else:
            output_images = [transforms.ToTensor()(img) for img in output_images]

        # Stack output images along the channel axis
        output_tensor = torch.cat(output_images, dim=0)

        return input_image, output_tensor

# Unet model
class UNET(nn.Module):
 
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Encoder part of unet
        self.encoder1 = self.conv_block(in_channels, 32)
        self.encoder2 = self.conv_block(32, 64)
        self.encoder3 = self.conv_block(64, 128)
        self.encoder4 = self.conv_block(128, 256)
        self.encoder5 = self.conv_block(256, 512)

        # bottleneck layer
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder part of unet
        self.upsampling5 = self.upsampling_block(1024, 512)
        self.decoder5 = self.conv_block(1024, 512)
        self.upsampling4 = self.upsampling_block(512, 256)
        self.decoder4 = self.conv_block(512, 256)
        self.upsampling3 = self.upsampling_block(256, 128)
        self.decoder3 = self.conv_block(256, 128)
        self.upsampling2 = self.upsampling_block(128, 64)
        self.decoder2 = self.conv_block(128, 64)
        self.upsampling1 = self.upsampling_block(64, 32)
        self.decoder1 = self.conv_block(64, 32)


        # changing to desired number of channels
        self.output = nn.Conv2d(32, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        conv =  nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

        return conv

    def forward(self, input):

        # Encoder part of unet
        encoder1 = self.encoder1(input)
        encoder2 = self.encoder2(nn.MaxPool2d(2)(encoder1))
        encoder3 = self.encoder3(nn.MaxPool2d(2)(encoder2))
        encoder4 = self.encoder4(nn.MaxPool2d(2)(encoder3))
        encoder5 = self.encoder5(nn.MaxPool2d(2)(encoder4))

        # bottleneck layer
        bottleneck = self.bottleneck(nn.MaxPool2d(2)(encoder5))

        # decoder part of unet
        decoder5 = self.upsampling5(bottleneck)
        decoder5 = torch.cat((decoder5, encoder5), dim=1)
        decoder5 = self.decoder5(decoder5)

        decoder4 = self.upsampling4(decoder5)
        decoder4 = torch.cat((decoder4, encoder4), dim=1)
        decoder4 = self.decoder4(decoder4)

        decoder3 = self.upsampling3(decoder4)
        decoder3 = torch.cat((decoder3, encoder3), dim=1)
        decoder3 = self.decoder3(decoder3)

        decoder2 = self.upsampling2(decoder3)
        decoder2 = torch.cat((decoder2, encoder2), dim=1)
        decoder2 = self.decoder2(decoder2)

        decoder1 = self.upsampling1(decoder2)
        decoder1 = torch.cat((decoder1, encoder1), dim=1)
        decoder1 = self.decoder1(decoder1)

        output = self.output(decoder1)
        return output

    def upsampling_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

class EnergyBasedLossFunction(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets, epsilon=1e-6):
        """
        predictions: Tensor of shape (B, N, T), predicted signals
        targets: Tensor of shape (B, N, T), ground truth signals
        epsilon: Small constant to avoid division by zero
        """
        # Compute MSE loss for each source in each sample
        mse_loss = torch.mean((predictions - targets) ** 2, dim=-1) # Shape: (B, N)

        # Compute energy for each source in each sample
        energies = torch.sum(targets ** 2, dim=-1) # Shape: (B, N)

        # Compute weights for each source in each sample
        weights = 1.0 / (energies + epsilon) # Shape: (B, N)

        # Compute weighted loss for each source in each sample
        weighted_losses = weights * mse_loss # Shape: (B, N)

        # Average over all sources and batch samples
        total_loss = torch.mean(weighted_losses) # Scalar

        return total_loss

# training
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 5 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# testing
def test(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()
    test_loss, correct_preds = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct_preds += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct_preds /= size
    print(
        f"Test Error: \n Accuracy: {correct_preds*100:>7f}%, Avg loss: {test_loss:>8f}"
    )

def predict_source_masks(model, spectrogram_image_path):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Loading the image
    img = Image.open(spectrogram_image_path).convert('L')
    
    # Creating a transformation pipeline
    transform = transforms.Compose([
    transforms.Resize((512, 512)),  
    transforms.ToTensor(),  # Convert image to tensor
    ])
    
    # Transforming the input image
    input_tensor = transform(img)
    
    # Adding batch dimension at the first position
    input_tensor = input_tensor.unsqueeze(0).to(device)
    
    # Model in evaluation model
    model.eval()
    
    with torch.no_grad():
        softmasks = model(input_tensor).detach().numpy()
    
    return softmasks

def stft(wavform, n_fft=1022, hop_length=512, window_length=1022):
    
    stft_results = torch.stft(wavform, n_fft=1022, hop_length=hop_length, win_length=window_length, window=torch.hann_window(window_length), return_complex=True)
    
    # Computing magnitude and phase
    magnitude = stft_results.abs()
    phase = torch.angle(stft_results)

    # Convert magnitude to decibels (log-compressed)
    magnitude_db = 20 * torch.log10(magnitude + 1e-6)

    # Normalize the magnitude spectrogram to range [0, 255] for grayscale
    magnitude_db_normalized = (magnitude_db - magnitude_db.min()) / (magnitude_db.max() - magnitude_db.min()) * 255
    magnitude_db_normalized = magnitude_db_normalized.squeeze().cpu().numpy().astype(np.uint8)
    
    magnitude_db_normalized = resample_spectrogram_db(magnitude_db_normalized, target_shape=(512, 512))
    resampled_phase = resample_spectrogram_phase(phase)
    
    return magnitude_db_normalized, resampled_phase

def istft(magnitude, phase, n_fft=1022, hop_length=512, window_length=1022):
    spec = magnitude * torch.exp(1j * torch.tensor(phase))
    return torch.istft(spec, n_fft=1022, hop_length=hop_length, win_length=window_length)

def separate_sources(mixed_audio_waveform, softmask, n_fft=1022, hop_length=512, window_length=1024):
    # Finding the magnitude and the phase of the mixed audio waveform
    mixed_magnitude, mixed_phase = stft(mixed_audio_waveform)
    
    # Multiplying the mixed audio waveform magnitude with the source mask
    masked_magnitude = torch.tensor(softmask) * torch.tensor(mixed_magnitude).unsqueeze(0)
    
    separated_sources = list()
    
    for index in range(softmask.shape[1]):  # Repeating this operation along the channel dimension
        source_magnitude = masked_magnitude[:, index, :, :]
        source_phase = mixed_phase 
        separate_source = istft(source_magnitude, source_phase)
        separated_sources.append(separate_source)
    
    # Save each separated source to a .wav file
    for i in range(len(separated_sources)):
        waveform = separated_sources[i]
        print(waveform.shape)
        waveform = waveform.cpu().numpy()  # Convert to numpy array
        
        # Normalize waveform to the range [-1, 1] for 16-bit PCM audio
        waveform = waveform.reshape(-1)  # Flatten to 1D if it's 1D already (samples,)
        waveform = np.clip(waveform, -1.0, 1.0)  # Normalize to [-1, 1]
        waveform = waveform.astype(np.float32)  # Convert to float32
        
        # Define sample rate (assuming it's 92160)
        sr = 10880
        
        # Ensure the 'Outputs' directory exists
        if not os.path.exists('Outputs'):
            os.makedirs('Outputs')
        
        instruments = ['Bass', 'Drums', 'Guitar', 'Piano', 'Others']
        
        # Save the waveform to a .wav file
        output_file = os.path.join('Outputs', f"waveform_{instruments[i]}.wav")  # Ensure .wav extension
        sf.write(output_file, waveform, sr)
        print(f"Saved waveform {i+1} to {output_file}")
        
    return separated_sources

if __name__ == "__main__":
    
    # cpu or cuda device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Paths and dimensions
    input_dir = os.path.join('Final_Dataset', 'Input')
    output_dir = os.path.join('Final_Dataset','Output')
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Create dataset
    dataset = UNetDataset(input_dir, output_dir, transform=transform)

    # DataLoader for batching and shuffling
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    
    # Checking input and output shapes
    for inputs, outputs in dataloader:
        print(f"Input shape: {inputs.shape}, Output shape: {outputs.shape}")
        break
    
    # Initializing the model
    in_channels, out_channels = 1, 5
    model = UNET(in_channels, out_channels).to(device)
    print(f"Model Architecture: \n{model}")
    
    # Loss and optimizer for training the model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    loss_fn = EnergyBasedLossFunction()
    
    # Training 
    epochs = 5
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------")
        train(dataloader, model, loss_fn, optimizer)
    
    # Saving the trained model
    if not os.path.exists('Models'):
        os.makedirs('Models')
    
    torch.save(model.state_dict(), os.path.join('Models', 'model_weights.pth'))
    
    # # Source separation
    # softmasks = predict_source_masks(model, os.path.join('Final_Dataset', 'Input', 'Track00001_mix.png'))
    # y, sr = librosa.load(os.path.join('RawData', 'Track00005', 'mix.wav'), mono=True, sr=10880)
    # y = make_lengths_same(y, sr)
    # separated_sources = separate_sources(torch.tensor(y, dtype=torch.float32), softmasks)
    
    
    