
# Importing required libraries
import librosa
import numpy as np
import pandas as pd
import IPython.display as ipd
import soundfile as sf
import os
import torch, torchaudio
import matplotlib.pyplot as plt
import re
import cv2
import zipfile
from PIL import Image
import shutil
import scipy.ndimage as ndimage

# Replacing all the instruments except in ['Piano', 'Drums', 'Bass', 'Guitar'] with 'Others' tag
def replace_other_track_labels(df, four_instr):
  try:
    df['Instrument Class'] = np.where(~df['Instrument Class'].isin(four_instr), 'Others', df['Instrument Class'])
    return df
  except Exception as e:
    print(f"Error encounterd in function 'replace_other_track_labels'.")
    raise e
  
# Making the length of every audio equal to 180 seconds
def make_lengths_same(audio_file, sample_rate=10880, target_duration=180):
    try:
        # Finding the length of input audio
        audio_length = len(audio_file)

        # Finding the target length in number of samples
        target_length = int(sample_rate * target_duration)

        if audio_length < target_length:  # If audio duration is less than 180 seconds
            padding = target_length - audio_length # Finding how much padding is required
            padding_left = 0  # Padding with zero
            padding_right = padding # Padding from the right side
            audio_file = np.pad(audio_file, (padding_left, padding_right), mode='constant', constant_values=0) # Padding

        elif audio_length > target_length: # If audio duration is greater than 180 seconds
            audio_file = audio_file[:target_length] # Cutting down the excess audio

        return audio_file
    except Exception as e:
        print(f"Error encounterd in the function 'make_lengths_same'.")
        raise e
      
def merge_tracks(track_df, instrument):

  try:
    # Filtering the track_df dataframe to get only the records with required instrument
    instr_df = track_df[track_df['Instrument Class'] == instrument]

    # If the required instrument is not present, return None
    if instr_df.shape[0] == 0:
      print(f"No matching instrument found for '{instrument}'")
      print("Available instruments:", track_df['Instrument Class'].unique())
      return None, None

    y = None
    sr = None

    # Iterating through each row of the filtered dataframe
    for index in range(instr_df.shape[0]):
      audio_path = os.path.join('RawData', instr_df.iloc[index, 0], 'stems', instr_df.iloc[index, 2])

      # Checking if the audio file exists
      if os.path.exists(audio_path):
        # If file exists, load it and proceed
        y_next, sr_next = librosa.load(audio_path, mono=True, sr=10880)
        y_next = make_lengths_same(y_next, sr_next)

        if y is None:  # If this is the first file for the instrument
          y = y_next
          sr = sr_next
        else:  # Add to the existing audio
          y += y_next

    if y is not None:
      # Normalizing the audio
      y /= np.max(np.abs(y))

    return y, sr

  except Exception as e:
    print("Error encounterd in the function 'merge_tracks'.")
    raise e
  
def merge_main_four_tracks(unique_track):

  try:
    # Filtering the data for the particular track
    track_df = df[df['Folder Name'] == unique_track]
    
    # Merging the piano records if required
    y_piano, sr_piano = merge_tracks(track_df, 'Piano')
    if y_piano is not None and sr_piano is not None :
      y_piano = y_piano.reshape(-1, 1) # reshaping because soundfile expects the shape of (audio_samples, num_channels)
    
    # Merging guitar records if required
    y_guitar, sr_guitar = merge_tracks(track_df, 'Guitar')
    if y_guitar is not None and sr_guitar is not None:
      y_guitar = y_guitar.reshape(-1, 1) # reshaping because soundfile expects the shape of (audio_samples, num_channels)

    # Merging the bass records if required
    y_bass, sr_bass = merge_tracks(track_df, 'Bass')
    if y_bass is not None and sr_bass is not None:
      y_bass = y_bass.reshape(-1, 1) # reshaping because soundfile expects the shape of (audio_samples, num_channels)

    # Merging the drum records if required
    y_drums, sr_drums = merge_tracks(track_df, 'Drums')
    if y_drums is not None and sr_drums is not None:
      y_drums = y_drums.reshape(-1, 1) # reshaping because soundfile expects the shape of (audio_samples, num_channels)

    return y_piano, y_guitar, y_bass, y_drums, sr_piano, sr_guitar, sr_bass, sr_drums

  except Exception as e:
    print(f"Error encountered in the function 'merge_main_four_tracks': {e}")
    raise e

def create_audio_dataset(unique_tracks, four_instr=['Piano', 'Drums', 'Bass', 'Guitar', 'Others']):

  try:
    # If the folder to store the data is not present then it is created
    if not os.path.exists('Audio_Dataset'):
      os.makedirs('Audio_Dataset')
    
    # Creating input and output folder
    if not os.path.exists(os.path.join('Audio_Dataset', 'Input')):
      os.makedirs(os.path.join('Audio_Dataset', 'Input'))

    if not os.path.exists(os.path.join('Audio_Dataset', 'Output')):
      os.makedirs(os.path.join('Audio_Dataset', 'Output'))

    for unique_track in unique_tracks: # For every unique trackk
      track_df = df[df['Folder Name'] == unique_track] # Filtering the data base on the unique track
      if all(True for instr in four_instr if instr in track_df['Instrument Class']): # If all the four main instruments and at least one other instrument are present in the mixed audio

        # Merging the multiple audio files of same instrument if required
        y_piano, y_guitar, y_bass, y_drums, sr_piano, sr_guitar, sr_bass, sr_drums = merge_main_four_tracks(unique_track)

        # If there is not other instrument present other than the main four then dummy others audio is created.
        if 'Others' not in track_df.iloc[:, 3].unique():
          y_others = np.zeros(int(180 * 10880)).reshape(-1, 1)
          sr_others = 10880
        else:
          y_others, sr_others = merge_tracks(track_df, 'Others')
            
        # Saving all four main audio files
        if y_piano is not None and y_drums is not None and y_bass is not None and y_guitar is not None and y_others is not None:
           # Creating a folder for each unique track if it is not already present
          if not os.path.exists(os.path.join('Audio_Dataset', 'Output', str(unique_track))):
            os.makedirs(os.path.join('Audio_Dataset', 'Output' ,str(unique_track)))
            
          sf.write(os.path.join('Audio_Dataset', 'Output',  str(unique_track), 'Piano.wav'), y_piano, sr_piano)
          sf.write(os.path.join('Audio_Dataset', 'Output' , str(unique_track), 'Drum.wav'), y_drums, sr_drums)
          sf.write(os.path.join('Audio_Dataset', 'Output' , str(unique_track), 'Bass.wav'), y_bass, sr_bass)
          sf.write(os.path.join('Audio_Dataset', 'Output' , str(unique_track), 'Guitar.wav'), y_guitar, sr_guitar)

          # Saving others audio
          if y_others is not None and sr_others is not None:
            sf.write(os.path.join('Audio_Dataset', 'Output' , str(unique_track), 'Others.wav'), y_others, sr_others)

          # Saving the mixed audio
          y_mix, sr_mix = librosa.load(os.path.join('RawData', str(unique_track), 'mix.wav'), mono=True, sr=10880)
          y_mix = make_lengths_same(y_mix, sr_mix)
          
          if y_mix is not None and sr_mix is not None:
            sf.write(os.path.join('Audio_Dataset', 'Input', f'{unique_track}_mix.wav'), y_mix, sr_mix)

  except Exception as e:
    print("Error encountered in the 'create_dataset' function.")
    
def resample_spectrogram_db(spectrogram, target_shape=(512, 512)):
    return ndimage.zoom(spectrogram, (target_shape[0] / spectrogram.shape[0], target_shape[1] / spectrogram.shape[1]), order=3)
  
def resample_spectrogram_phase(phase, target_shape=(512, 512)):
    return ndimage.zoom(phase, (target_shape[0] / phase.shape[0], target_shape[1] / phase.shape[1]), order=3)
  
def create_log_magnitude_spectrogram(waveform, window_length=1022, hop_length=512, sample_rate=10880):
    # Ensure waveform is a torch tensor
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.tensor(waveform, dtype=torch.float32)

    # Ensure correct shape: (channels, samples)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # Add channel dimension if missing

    # Validate tensor dimensions
    if waveform.ndim != 2:
        raise ValueError(f"Waveform must be 2D (channels, samples), got shape {waveform.shape}")

    stft_results = torch.stft(waveform, n_fft=1022, hop_length=hop_length, win_length=window_length, window=torch.hann_window(window_length), return_complex=True)

    # Computing magnitude and phase
    magnitude = stft_results.abs()
    phase = torch.angle(stft_results)

    # # Save magnitude and phase
    # if not os.path.exists('Magnitude_Phase_Info'):
    #     os.makedirs('Magnitude_Phase_Info')

    # torch.save(magnitude, os.path.join('Magnitude_Phase_Info', 'Mixture_magnitude.pt'))
    # torch.save(phase, os.path.join('Magnitude_Phase_Info', 'Mixture_phase.pt'))

    # Convert magnitude to decibels (log-compressed)
    magnitude_db = 20 * torch.log10(magnitude + 1e-6)

    # Normalize the magnitude spectrogram to range [0, 255] for grayscale
    magnitude_db_normalized = (magnitude_db - magnitude_db.min()) / (magnitude_db.max() - magnitude_db.min()) * 255
    magnitude_db_normalized = magnitude_db_normalized.squeeze().cpu().numpy().astype(np.uint8)
    
    magnitude_db_normalized = resample_spectrogram_db(magnitude_db_normalized, target_shape=(513, 513))
    return magnitude_db_normalized

def create_spectrogram_dataset(unique_tracks, four_instr=['Piano', 'Drums', 'Bass', 'Guitar', 'Others']):
    try:
        # If the folder to store the data is not present then it is created
        if not os.path.exists('Spectrogram_Dataset'):
            os.makedirs('Spectrogram_Dataset')

        # Creating input and output folder
        if not os.path.exists(os.path.join('Spectrogram_Dataset', 'Input')):
            os.makedirs(os.path.join('Spectrogram_Dataset', 'Input'))

        if not os.path.exists(os.path.join('Spectrogram_Dataset', 'Output')):
            os.makedirs(os.path.join('Spectrogram_Dataset', 'Output'))

        for unique_track in unique_tracks: # For every unique trackk
            track_df = df[df['Folder Name'] == unique_track] # Filtering the data base on the unique track
            if all(True for instr in four_instr if instr in track_df['Instrument Class']):  # If all the four main instruments are present in the mixed audio

                # Merging the multiple audio files of same instrument if required
                y_piano, y_guitar, y_bass, y_drums, sr_piano, sr_guitar, sr_bass, sr_drums = merge_main_four_tracks(unique_track)

                # If there is not other instrument present other than the main four then dummy others audio is created.
                if 'Others' not in track_df.iloc[:, 3].unique():
                    y_others = np.zeros(int(180 * 10880)).reshape(-1, 1)
                    sr_others = 10880
                else:
                    y_others, sr_others = merge_tracks(track_df, 'Others')
                
                if y_piano is not None and y_drums is not None and y_bass is not None and y_guitar is not None and y_others is not None:
                    # Creating a folder for each unique track if it is not already present
                    if not os.path.exists(os.path.join('Spectrogram_Dataset', 'Output', str(unique_track))):
                        os.makedirs(os.path.join('Spectrogram_Dataset', 'Output' ,str(unique_track)))

                    y_mix, sr_mix = librosa.load(os.path.join('RawData', str(unique_track), 'mix.wav'), mono=True, sr=10880)

                    # Defining the parameters for the mel spectrogram
                    window_length = 1022
                    hop_length = 512
                    sample_rate = 10880

                    # for input audio
                    # performing short time fourier transform (STFT) with hanning window
                    y_mix = y_mix.reshape(1, -1) # torchaudio needs the shape (num_channels, num_samples)

                    input_log_magnitude_spectrogram_db = create_log_magnitude_spectrogram(y_mix, window_length, hop_length, sample_rate)

                    # plotting and saving the mel-spectrogram
                    fig = plt.figure(figsize=(7,7))
                    cax = plt.imshow(input_log_magnitude_spectrogram_db, aspect='auto', origin='lower', interpolation=None,  cmap='viridis')
                    # plt.colorbar(format='%+2.0f dB')
                    cbar = plt.colorbar(cax)
                    cbar.remove()
                    plt.tight_layout()
                    plt.savefig(os.path.join('Spectrogram_Dataset', 'Input', f"{unique_track}_mix.png"))
                    plt.close(fig)

                    outputs = [y_piano, y_guitar, y_bass, y_drums, y_others]
                    instr_names = ['Piano', 'Guitar', 'Bass', 'Drums', 'Others']
                    # for output audios
                    for index, output in enumerate(outputs):
                        if output is not None:
                            output = output.reshape(1, -1) # torchaudio needs the shape (num_channels, num_samples)
                            output_mel_spectrogram_db = create_log_magnitude_spectrogram(output, window_length, hop_length, sample_rate)
        
                            # plotting and saving the spectrogram
                            fig = plt.figure(figsize=(7, 7))
                            cax = plt.imshow(output_mel_spectrogram_db, aspect='auto', origin='lower', interpolation=None,  cmap='viridis')
                            plt.axis('off')
                            # plt.colorbar(format='%+2.0f dB')
                            cbar = plt.colorbar(cax)
                            cbar.remove()
                            plt.tight_layout()
                            plt.savefig(os.path.join('Spectrogram_Dataset', 'Output', str(unique_track), f"{instr_names[index]}.png"))
                            plt.close(fig)

    except Exception as e:
        print("Error encountered in the function 'create_spectrogram_dataset'.")
        raise e

def load_spectrogram_image(image_path):
    # Opening the image using PIL and convert to grayscale
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    return img_array
  
def create_mask_dataset():
    
    output_dirs = os.listdir(os.path.join('Spectrogram_Dataset', 'Output'))

    for output_dir in output_dirs:

        output_dir_path = os.path.join('Spectrogram_Dataset', 'Output', output_dir)
        source_images = os.listdir(output_dir_path)

        source_img_array = [load_spectrogram_image(os.path.join(output_dir_path, source_image)) for source_image in source_images]
        
        # Calculate the sum of all sources' magnitudes at each time-frequency point
        magnitude_sum = np.sum(source_img_array, axis=0)  # along the dimension of sources

        # Computing the soft masks
        epsilon = 1e-10
        magnitude_sum = np.maximum(magnitude_sum, epsilon) # ensuring no zero values are present in the sum

        softmasks = source_img_array / magnitude_sum

        if not os.path.exists('Final_Dataset'):
            os.makedirs('Final_Dataset')

        if not os.path.exists(os.path.join('Final_Dataset', 'Output')):
            os.makedirs(os.path.join('Final_Dataset', 'Output'))

        for index, softmask in enumerate(softmasks):

            if not os.path.exists(os.path.join('Final_Dataset', 'Output', output_dir)):
                os.makedirs(os.path.join('Final_Dataset', 'Output', output_dir))

            fig = plt.figure(figsize=(7,7))
            plt.axis('off')
            plt.imshow(softmask, cmap='gray', origin='lower', aspect='auto')
            plt.savefig(os.path.join('Final_Dataset', 'Output', output_dir, source_images[index]), bbox_inches='tight', transparent=True)
            plt.close(fig)
  
if __name__ == "__main__":
    
  # Reading the file
  df = pd.read_csv('slakh2100_metadata.csv')
  
  # Four main instruments
  four_instr = ['Piano', 'Drums', 'Bass', 'Guitar']

  # Unique track folders
  unique_tracks = df['Folder Name'].unique()
  
  # Replacing all the instruments except in ['Piano', 'Drums', 'Bass', 'Guitar'] with 'Others' tag in csv file
  df = replace_other_track_labels(df, four_instr)
  
  # create audio dataset
  create_audio_dataset(unique_tracks)
  
  # create spectrogram dataset
  create_spectrogram_dataset(unique_tracks, four_instr=['Piano', 'Drums', 'Bass', 'Guitar'])
  
  # Creating final dataset i.e., input --> spectrogram, output --> softmasks
  
  # Copying input sepctrograms to the final dataset folder
  source_folder = os.path.join('Spectrogram_Dataset', 'Input')
  destination_folder = os.path.join('Final_Dataset', 'Input')
  shutil.copytree(source_folder, destination_folder)
  
  create_mask_dataset()
  
  
  