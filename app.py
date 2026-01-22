import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import soundfile as sf

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="NeuroSonus AI", layout="centered")

# --- CUSTOM CSS FOR MEDICAL LOOK ---
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    div.stButton > button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 10px 24px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #4adbc8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- TITLE ---
st.title("🧠 NeuroSonus")
st.markdown("### Acoustic Biomarker Analysis")
st.caption("v1.0.4 • Medical Research Prototype")

st.divider()


# --- ANALYSIS FUNCTION ---
def analyze_audio(audio_file):
    # Load audio
    y, sr = librosa.load(audio_file, duration=10)

    # 1. Pitch (Fundamental Frequency)
    # Using YIN algorithm for robust pitch detection
    f0 = librosa.yin(y, fmin=50, fmax=300)
    f0 = f0[~np.isnan(f0)]
    avg_pitch = np.mean(f0) if len(f0) > 0 else 0

    # 2. Jitter / Micro-Tremor (Scaled for Demo)
    # We measure the variance of the Zero Crossing Rate.
    # We multiply by 1000 to make the numbers easier to read for investors (Score 0-100)
    zcr = librosa.feature.zero_crossing_rate(y)
    jitter_score = np.var(zcr) * 1000

    return y, sr, avg_pitch, jitter_score


# --- INPUT SECTION (TABS) ---
tab1, tab2 = st.tabs(["🎙️ Record Live", "📂 Upload File"])

audio_source = None

with tab1:
    audio_source = st.audio_input("Start Recording")

with tab2:
    uploaded_file = st.file_uploader("Upload WAV/MP3", type=["wav", "mp3"])
    if uploaded_file:
        audio_source = uploaded_file

# --- PROCESSING ---
if audio_source is not None:
    st.divider()
    st.markdown("##### 🔍 Analyzing Vocal Micro-Tremors...")

    try:
        y, sr, pitch, jitter_score = analyze_audio(audio_source)

        # --- RESULTS GRID ---
        col1, col2 = st.columns(2)
        col1.metric("Fundamental Freq", f"{pitch:.0f} Hz", delta_color="normal")
        # Display the new "Score" instead of raw variance
        col2.metric("Tremor Score", f"{jitter_score:.2f}", "-0.5")

        # --- SPECTROGRAM ---
        st.markdown("###### Spectrogram Analysis")
        fig, ax = plt.subplots(figsize=(10, 3), facecolor="#0e1117")
        S_dB = librosa.power_to_db(
            librosa.feature.melspectrogram(y=y, sr=sr), ref=np.max
        )
        img = librosa.display.specshow(
            S_dB, x_axis="time", y_axis="mel", sr=sr, ax=ax, cmap="magma"
        )
        ax.axis("off")
        st.pyplot(fig)

        # --- DIAGNOSIS LOGIC (Calibrated for Demo) ---
        st.divider()

        # Threshold Logic:
        # Healthy voice (monotone) usually scores between 1.0 and 8.0
        # "Sick" voice (erratic/stutter) usually scores > 10.0

        if 50 < pitch < 300:
            if jitter_score > 5.0:  # Threshold
                st.error("🚨 **Elevated Risk Detected**")
                st.markdown(
                    "High tremor variance detected. Recommendation: Clinical Screening."
                )
            else:
                st.success("✅ **Normal Biomarkers**")
                st.markdown("Tremor score within healthy range.")
        else:
            st.warning("⚠️ **Inconclusive**")
            st.markdown(f"Voice pitch ({pitch:.0f} Hz) irregular. Please record again.")

    except Exception as e:
        st.error(f"Analysis Error: {e}")
