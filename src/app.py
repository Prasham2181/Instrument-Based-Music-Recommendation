import streamlit as st
import os
import tempfile
import librosa
import numpy as np
from ModelTraining import predict_source_masks, separate_sources, make_lengths_same
from DatasetLoading import create_log_magnitude_spectrogram
import torch
import matplotlib.pyplot as plt
from ModelTraining import UNET

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


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

    return duration_matrix

def generate_recommendations():
    
    instrument_durations = calculate_instrument_durations()
    print(f"**********instrument duration: {instrument_durations}")
    db_durations = np.load('db_duration_matrix.npy')
    print(f"**********db duration: {db_durations}")
    cosine_similarity = calculate_similarity_score(instrument_durations, db_durations)
    
    max_index = np.argmax(cosine_similarity)
    song_options = sorted(os.listdir(os.path.join('Audio_Dataset', 'test', 'Input')))
    
    recommendations_file_name = song_options[max_index]
    
    return recommendations_file_name

def calculate_similarity_score(instrument_durations, db_durations):
    
    # Using similarity score formula
    instrument_durations = np.array(instrument_durations).reshape(-1, 1)
    
    # magnitudes of two array
    instrument_durations_magnitude = np.linalg.norm(instrument_durations)
    db_durations_magnitude = np.linalg.norm(db_durations, axis=0)
    
    cosine_similarity = np.dot(db_durations, instrument_durations) / (instrument_durations_magnitude * db_durations_magnitude)
    print(f"********cosine similarity: {cosine_similarity}")

    return cosine_similarity

def webapp():
    # # Streamlit UI Part

    st.set_page_config(
            page_title = "Music Recommendation App",
            page_icon="🎹",
            initial_sidebar_state="expanded",
            # layout='wide'
        )
    
    st.markdown("""
    <style>
        /* Ensure the full width usage for the overall container */
        .reportview-container {
            max-width: 100% !important;
        }

        /* Center the content container and set it to 60% width */
        .block-container {
            max-width: 60%;    /* Set the content width to 60% of the screen */
            margin: 0 auto;    /* Center the container horizontally */
            padding-left: 20px !important;
            padding-right: 20px !important;
        }

        /* Apply full width to main content section */
        .main {
            max-width: 100% !important;
        }

        /* Center all headers (h1, h2, etc.) */
        h1, h2, h3, h4, h5, h6 {
            text-align: center;
        }

        /* Center all text content */
        .stMarkdown, .stText {
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("Instrument-Based Music Recommendation 🎹")

    st.caption('A project by Shivam Shinde, Prasham Soni, and Manav Mepani')

    st.image('Header_Image.jpg')

    st.divider()

    st.subheader('Upload a song file')
    uploaded_file = st.file_uploader("", type=["wav"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_file_path = temp_file.name
            
        if not os.path.exists('user_ip_wavfile_folder'):
            os.makedirs('user_ip_wavfile_folder')
            
        file_path = os.path.join('user_ip_wavfile_folder', 'wavfile.wav')
        
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        st.audio(temp_file_path)

        st.subheader("Tell us your instrument preferences:")
        guitar_pref = st.radio("Do you prefer Guitar:guitar: in recommendations?", ["No", "Yes"])
        drums_pref = st.radio("Do you prefer Drums:drum_with_drumsticks: in recommendations?", ["No", "Yes"])
        piano_pref = st.radio("Do you prefer Piano:musical_keyboard: in recommendations?", ["No", "Yes"])
        bass_pref = st.radio("Do you prefer Bass:notes: in recommendations?", ["No", "Yes"])
        others_pref = st.radio("Do you prefer other:musical_score: instruments in recommendations?", ["No", "Yes"])

        preferences = [
            1 if guitar_pref == "Yes" else 0,
            1 if drums_pref == "Yes" else 0,
            1 if piano_pref == "Yes" else 0,
            1 if bass_pref == "Yes" else 0,
            1 if others_pref == "Yes" else 0,
        ]    
        

        if st.button("Submit"):
            with st.spinner():
                recommendations = generate_recommendations()
                print(f"******************Recommendations: {recommendations}")
                if recommendations:
                    st.success("Preferences submitted successfully! Here is your recommendation:")
                    # for rec in recommendations:
                    #     st.write(f"- {rec}")
                    st.audio(os.path.join('Database', 'Input', f"{recommendations}"))
                else:
                    st.warning("No recommendations available based on your preferences. Try adjusting your inputs.")

        try:
            os.remove(temp_file_path)
        except Exception as e:
            st.error(f"Error cleaning up the file: {e}")
    else:
        st.info("Please upload a song to proceed.")
    
    
if __name__ == "__main__":

    webapp()


