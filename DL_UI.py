import streamlit as st
import os
import tempfile

def calculate_instrument_weights(song_path):
#    random values taken at the moment
    return [1, 3, 2, 4, 5]  # Guitar, Drums, Piano, Bass, Others

def generate_recommendations(weights):
    # random values of song a b c taken 
    return ["Song A", "Song B", "Song C"] if sum(weights) > 0 else []

st.title("CS 541 Deep Learning Final Project")

st.header("Instrument-Based Music Recommendation 🎹")

user_name = st.text_input("Enter your name:", "")
if user_name:
    st.subheader(f"Hi {user_name}\n")
    st.header(f"Welcome to the Song Recommendation App")


uploaded_file = st.file_uploader("Upload a song file (.mp3 or .wav):", type=["mp3", "wav"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_file_path = temp_file.name

    st.audio(temp_file_path)
    
    instrument_weights = calculate_instrument_weights(temp_file_path)

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

    selected_weights = [instrument_weights[i] * preferences[i] for i in range(5)]

    if st.button("Submit"):
        recommendations = generate_recommendations(selected_weights)
        if recommendations:
            st.success("Preferences submitted successfully! Here are your recommendations:")
            for rec in recommendations:
                st.write(f"- {rec}")
        else:
            st.warning("No recommendations available based on your preferences. Try adjusting your inputs.")

    try:
        os.remove(temp_file_path)
    except Exception as e:
        st.error(f"Error cleaning up the file: {e}")
else:
    st.info("Please upload a song to proceed.")
