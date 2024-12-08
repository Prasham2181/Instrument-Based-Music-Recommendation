import os
import librosa
import numpy as np

# Function to calculate non-silent duration percentage
def get_instrument_duration_percentage(file_path):
    """
    Calculate the percentage duration of non-silent segments in an audio file.

    Parameters:
    - file_path: Path to the audio file.

    Returns:
    - Non-silent duration as a percentage of the total duration.
    """
    # Load the audio file
    y, sr = librosa.load(file_path, sr=None)

    # Identify non-silent intervals (using top_db=20 as the threshold)
    intervals = librosa.effects.split(y, top_db=20)

    # Calculate the total duration of non-silent segments
    non_silent_duration = np.sum([(end - start) / sr for start, end in intervals])

    # Total duration of the audio file
    total_duration = librosa.get_duration(y=y, sr=sr)

    # Avoid division by zero
    if total_duration == 0:
        return 0.0

    # Return the percentage of non-silent duration
    return (non_silent_duration / total_duration) * 100

# Define the path to the test folder
test_folder = r'C:\Users\sonip\Desktop\Playing with Pytorch\DL_Project\Track00001'

# Define the instruments (order matters for row assignment)
instruments = ['Guitar', 'Bass', 'Piano', 'Drum', 'Others']

# Initialize an empty dictionary to store durations for each track
duration_matrix = []

# Iterate over each track folder
for track_folder in sorted(os.listdir(test_folder)):
    track_path = os.path.join(test_folder, track_folder)
    if not os.path.isdir(track_path):
        continue  # Skip files, only process directories

    # Initialize a list to store durations for the current track
    track_durations = []

    # Iterate over each instrument
    for instrument in instruments:
        instrument_file = os.path.join(track_path, f"{instrument}.wav")
        if os.path.exists(instrument_file):
            # Calculate the percentage of non-silent audio for the instrument
            duration_percentage = get_instrument_duration_percentage(instrument_file)
        else:
            # If the instrument file is missing, set percentage to 0
            duration_percentage = 0.0

        # Append the percentage to the current track's list
        track_durations.append(duration_percentage)

    # Append the track's percentages to the matrix
    duration_matrix.append(track_durations)

# Convert the matrix to a NumPy array for saving
duration_matrix = np.array(duration_matrix)

# Save the matrix as a .npy file
output_npy = r'C:\Users\sonip\Desktop\Playing with Pytorch\DL_Project\instrument_durations_percentage.npy'
np.save(output_npy, duration_matrix)

# Print confirmation
print(f"Duration matrix saved as .npy file at: {output_npy}")

# Optional: Print the matrix for verification
print("Duration Matrix (rows: tracks, columns: instruments):")
print(duration_matrix)
