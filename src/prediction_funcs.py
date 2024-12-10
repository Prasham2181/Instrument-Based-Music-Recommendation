from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import soundfile as sf
import os
import logging
import json
from step0_utility_functions import Utility
import numpy as np
from step2_DatasetLoading import DataLoadingProcessing

class Predictions:

    def predict_source_masks(self, model, spectrogram_image_path):
        
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

    def stft(self, wavform, n_fft=1022, hop_length=512, window_length=1022):
        
        stft_results = torch.stft(wavform, n_fft=1022, hop_length=hop_length, win_length=window_length, window=torch.hann_window(window_length), return_complex=True)
        
        # Computing magnitude and phase
        magnitude = stft_results.abs()
        phase = torch.angle(stft_results)

        # Convert magnitude to decibels (log-compressed)
        magnitude_db = 20 * torch.log10(magnitude + 1e-6)

        # Normalize the magnitude spectrogram to range [0, 255] for grayscale
        magnitude_db_normalized = (magnitude_db - magnitude_db.min()) / (magnitude_db.max() - magnitude_db.min()) * 255
        magnitude_db_normalized = magnitude_db_normalized.squeeze().cpu().numpy().astype(np.uint8)
        
        magnitude_db_normalized = DataLoadingProcessing().resample_spectrogram_db(magnitude_db_normalized, target_shape=(512, 512))
        resampled_phase = DataLoadingProcessing().resample_spectrogram_phase(phase)
        
        return magnitude_db_normalized, resampled_phase

    def istft(self, magnitude, phase, n_fft=1022, hop_length=512, window_length=1022):
        spec = magnitude * torch.exp(1j * torch.tensor(phase))
        return torch.istft(spec, n_fft=1022, hop_length=hop_length, win_length=window_length)

    def separate_sources(self, mixed_audio_waveform, softmask, n_fft=1022, hop_length=512, window_length=1024):
        # Finding the magnitude and the phase of the mixed audio waveform
        mixed_magnitude, mixed_phase = self.stft(mixed_audio_waveform)
        
        # Multiplying the mixed audio waveform magnitude with the source mask
        masked_magnitude = torch.tensor(softmask) * torch.tensor(mixed_magnitude).unsqueeze(0)
        
        separated_sources = list()
        
        for index in range(softmask.shape[1]):  # Repeating this operation along the channel dimension
            source_magnitude = masked_magnitude[:, index, :, :]
            source_phase = mixed_phase 
            separate_source = self.istft(source_magnitude, source_phase)
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
