import os
import librosa
import numpy as np
from step4_ModelTraining import predict_source_masks, separate_sources, make_lengths_same
from step2_DatasetLoading import create_log_magnitude_spectrogram
import torch
import matplotlib.pyplot as plt
from step4_ModelTraining import UNET
import logging
from step0_utility_functions import Utility



def get_instrument_duration(file_path):

  # load audio file
  y, sr = librosa.load(file_path)

  # identity non-silent intervals
  intervals = librosa.effects.split(y, top_db=20)

  # calculate total duration of non-silent segment
  duration = np.sum([(end - start) / sr for start, end in intervals])

  # Total duration of an audio
  total_duration = librosa.get_duration(y=y, sr=sr)  
  
  return duration/total_duration

def calculate_instrument_durations(song_file_path=os.path.join('user_ip_wavfile_folder', 'wavfile.wav')):
    
    # Loading the trained model
    model = UNET(1, 5)
    state_dict = torch.load(os.path.join('Models', 'model_weights.pth'))
    model.load_state_dict(state_dict)
    
    # Finding the wavform and sample rate
    y, sr = librosa.load(song_file_path, mono=True, sr=10880)
    
    # Making length = 180 seconds
    y = make_lengths_same(y, sr)
    
    user_ip_spectrogram = create_log_magnitude_spectrogram(y, window_length=1022, hop_length=512, sample_rate=10880)
    
    if not os.path.exists('User_ip_spectrogram'):
        os.makedirs('User_ip_spectrogram')
    
    fig = plt.figure(figsize=(7,7))
    cax = plt.imshow(user_ip_spectrogram, aspect='auto', origin='lower', interpolation=None,  cmap='viridis')
    cbar = plt.colorbar(cax)
    cbar.remove()
    plt.tight_layout()
    plt.savefig(os.path.join('User_ip_spectrogram', f"user_ip_spectrogram.png"))
    plt.close(fig)
    
    # Predicting the softmask of sources
    softmasks = predict_source_masks(model, os.path.join('User_ip_spectrogram', f"user_ip_spectrogram.png"))
    
    # Separating sources i.e., saving separated sources into wavforms
    separated_sources = separate_sources(torch.tensor(y, dtype=torch.float32), softmasks)
    
    # Calculating the durations of sources
    durations = list()
    source_files = os.listdir(os.path.join('Outputs'))
    
    for source in source_files:
        path = os.path.join('Outputs', source)
        durations.append(get_instrument_duration(path))
        
    return durations  # Guitar, Drums, Piano, Bass, Others

def calculate_db_durations(test_folder=os.path.join('Audio_Dataset', 'test', 'Output'), instruments= ['Bass', 'Drums', 'Guitar', 'Piano', 'Others'], output_file_path='db_duration_matrix.npy'):
    # Initialize the matrix
    duration_matrix = []
    
    # Process each track folder
    for track_folder in sorted(os.listdir(test_folder)):
        track_path = os.path.join(test_folder, track_folder)
        
        if not os.path.isdir(track_path):
            continue  # Skip files, only process directories

        # Calculate durations for the current track
        track_durations = []
        for instrument in instruments:
            instrument_file = os.path.join(track_path, f"{instrument}.wav")
            if os.path.exists(instrument_file):
                duration_percentage = get_instrument_duration(instrument_file)
            else:
                duration_percentage = 0.0
                
            track_durations.append(duration_percentage)

        # Append to the matrix
        duration_matrix.append(track_durations)

    duration_matrix = np.array(duration_matrix)
    np.save(output_file_path, duration_matrix)

    # return duration_matrix

def generate_recommendations():
    
    instrument_durations = calculate_instrument_durations()
    logger.info('Instrument durations calculated.')

    # print(f"**********instrument duration: {instrument_durations}")
    db_durations = np.load('db_duration_matrix.npy')
    logger.info('Database similarity matrix loaded.')
    # print(f"**********db duration: {db_durations}")
    cosine_similarity = calculate_similarity_score(instrument_durations, db_durations)
    logger.info('Cosine similarity between instrument duration array and database durations matrix.')
    max_index = np.argmax(cosine_similarity)
    song_options = sorted(os.listdir(os.path.join('Audio_Dataset', 'test', 'Input')))
    
    recommendations_file_name = song_options[max_index]
    logger.info('Determined the recommendation.')
    
    return recommendations_file_name

def calculate_similarity_score(instrument_durations, db_durations):
    
    # Using similarity score formula
    instrument_durations = np.array(instrument_durations).reshape(-1, 1)
    
    # magnitudes of two array
    instrument_durations_magnitude = np.linalg.norm(instrument_durations)
    db_durations_magnitude = np.linalg.norm(db_durations, axis=0)
    
    cosine_similarity = np.dot(db_durations, instrument_durations) / (instrument_durations_magnitude * db_durations_magnitude)
    # print(f"********cosine similarity: {cosine_similarity}")

    return cosine_similarity


if __name__ == "__main__":

    # SETTING UP THE LOGGING MECHANISM
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    Utility().create_folder('Logs')
    params = Utility().read_params()

    main_log_folderpath = params['Logs']['Logs_Folder']
    Make_Predictions = params['Logs']['make_predictions']

    file_handler = logging.FileHandler(os.path.join(
        main_log_folderpath, Make_Predictions))
    formatter = logging.Formatter(
        '%(asctime)s : %(levelname)s : %(filename)s : %(message)s')

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

  # STARTING THE EXECUTION OF FUNCTIONS

    calculate_db_durations()