import streamlit as st
import time
import json
import os
import asyncio
import edge_tts
from pydub import AudioSegment
from pydub.playback import play
import speech_recognition as sr
import sounddevice as sd
from scipy.io.wavfile import write
from llm_scoring import score_all_responses
from datetime import datetime

# Set FFmpeg path (update this to your FFmpeg path)
AudioSegment.converter = "C:\ffmpeg\bin\ffmpeg.exe"  # Use this if ffmpeg is in system PATH
# Or specify full path if needed:
# AudioSegment.converter = "/path/to/ffmpeg"  # Linux/Mac
# AudioSegment.converter = "C:\\path\\to\\ffmpeg.exe"  # Windows

st.set_page_config(page_title="Online Interview System", layout="centered", initial_sidebar_state="collapsed")

hide_sidebar_and_toggle = """
    <style>
        [data-testid="stSidebar"] {
            display: none !important;
        }
        [data-testid="collapsedControl"] {
            display: none !important;
        }
    </style>
"""
st.markdown(hide_sidebar_and_toggle, unsafe_allow_html=True)

# Initialize session state variables
if 'candidate_id' not in st.session_state:
    st.session_state.candidate_id = ""
if 'current_q' not in st.session_state:
    st.session_state.current_q = 0
if 'responses' not in st.session_state:
    st.session_state.responses = []
if 'interview_started' not in st.session_state:
    st.session_state.interview_started = False
if 'start_record' not in st.session_state:
    st.session_state.start_record = False
if 'id_confirmed' not in st.session_state:
    st.session_state.id_confirmed = False
if 'terminate_clicked' not in st.session_state:
    st.session_state.terminate_clicked = False

# Sample questions
questions = [
    "What is a process in an operating system?",
    "What is a system call in OS?",
    "Explain the concept of virtual memory.",
    "What is a deadlock? How can it be prevented?",
    "Describe the function of a kernel."
]


# TTS function using edge_tts
def speak(text):
    try:
        output_file = "question.mp3"
        # Run async TTS in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        communicate = edge_tts.Communicate(text, "en-US-EricNeural")
        loop.run_until_complete(communicate.save(output_file))
        loop.close()

        if os.path.exists(output_file):
            sound = AudioSegment.from_mp3(output_file)
            play(sound)
            os.remove(output_file)
        else:
            st.error("TTS output file not created")
    except Exception as e:
        st.error(f"TTS error: {e}")


def play_beep():
    try:
        # Generate beep sound programmatically
        beep = AudioSegment.silent(duration=100).append(
            AudioSegment.from_tones([800], duration=200, volume=-10),
            crossfade=0
        )
        play(beep)
    except Exception as e:
        st.warning(f"Beep generation failed: {e}")


# Audio recording function
def record_audio(filename, duration=30, fs=44100):
    st.info("Please speak clearly.")
    timer_placeholder = st.empty()
    try:
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        for remaining in range(duration, 0, -1):
            timer_placeholder.markdown(f"⏳ **Time left: {remaining} seconds**")
            time.sleep(1)
        sd.wait()
        write(filename, fs, audio)
        timer_placeholder.markdown("**Oops! Time's up. ⏰**")
        return filename
    except Exception as e:
        st.error(f"Recording failed: {e}")
        return None


# Audio transcription function
def transcribe_audio(filename):
    if not filename or not os.path.exists(filename):
        return "No valid audio file."
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(filename) as source:
            audio = recognizer.record(source)
            return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Could not understand audio."
    except sr.RequestError as e:
        return f"API error: {e}"
    except Exception as e:
        return f"Transcription error: {e}"


st.title("Online Interview System")

# Candidate ID confirmation
if not st.session_state.id_confirmed:
    candidate_id_input = st.text_input("Enter your Candidate ID to begin:")
    if candidate_id_input:
        if st.button("Confirm ID"):
            st.session_state.candidate_id = candidate_id_input
            st.session_state.id_confirmed = True
            st.rerun()
    st.stop()

# Interview start
if not st.session_state.interview_started:
    st.success(f"ID '{st.session_state.candidate_id}' registered. You may start the interview.")
    if st.button("Start Interview"):
        st.session_state.interview_started = True
        st.session_state.start_record = False
        st.rerun()
    st.stop()

# Create candidate directory
candidate_dir = os.path.join("interviews", st.session_state.candidate_id)
os.makedirs(candidate_dir, exist_ok=True)

# Interview process
q_idx = st.session_state.current_q

if q_idx < len(questions) and not st.session_state.terminate_clicked:
    question = questions[q_idx]
    st.subheader(f"Question {q_idx + 1}: {question}")

    # Audio flag for question playback
    audio_flag_key = f"audio_played_{q_idx}"
    if audio_flag_key not in st.session_state:
        st.session_state[audio_flag_key] = False

    # Play question audio
    if not st.session_state[audio_flag_key]:
        if st.button("▶️ Play Question", key=f"play_q{q_idx}"):
            play_beep()
            speak(question)
            play_beep()
            st.session_state[audio_flag_key] = True
            st.session_state.start_record = True
            st.rerun()

    # Record answer
    if st.session_state.start_record:
        audio_filename = os.path.join(candidate_dir, f"q{q_idx + 1}_answer.wav")
        audio_path = record_audio(audio_filename)
        if audio_path:
            play_beep()
            transcript = transcribe_audio(audio_path)

            st.session_state.responses.append({
                "question_number": q_idx + 1,
                "question": question,
                "audio_file": audio_filename,
                "transcript": transcript
            })

            st.session_state.current_q += 1
            st.session_state.start_record = False
            st.rerun()

# Interview control buttons
if not st.session_state.terminate_clicked:
    if st.session_state.current_q >= len(questions):
        st.warning("🎉 You've answered all questions. Click **View My Responses** to view the summary and submit.")
        if st.button("🟢 View My Responses", key="terminate_btn_final"):
            st.session_state.terminate_clicked = True
            st.rerun()
    elif len(st.session_state.responses) > 0:
        if st.button("🔴 Quit Interview", key="terminate_btn"):
            st.session_state.terminate_clicked = True
            st.rerun()

# Show responses after interview completion
if st.button("Submit and Show Results"):
    try:
        # Save responses
        candidate_dir = os.path.join("interviews", st.session_state.candidate_id)
        os.makedirs(candidate_dir, exist_ok=True)

        with open(os.path.join(candidate_dir, "responses.json"), "w") as f:
            json.dump(st.session_state.responses, f, indent=4)

        # Score responses with a spinner in the main app
        with st.spinner("Evaluating your responses. This may take a minute..."):
            scoring_results = score_all_responses(st.session_state.candidate_id)

        # Store in session state
        st.session_state.scoring_results = scoring_results

        # Redirect to thank you page
        st.success("Evaluation complete! Redirecting...")
        time.sleep(1)
        st.switch_page("pages/thank_you.py")

    except Exception as e:
        st.error(f"Scoring failed: {e}")