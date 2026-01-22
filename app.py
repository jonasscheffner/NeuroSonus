import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import soundfile as sf

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="NeuroSonus AI", layout="centered")

# --- CUSTOM CSS ---
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
st.caption("v1.0.5 • Tremor Stability Algorithm")

st.divider()


# --- ANALYSIS FUNCTION ---
def analyze_audio(audio_file):
    # Load audio
    y, sr = librosa.load(audio_file, duration=10)

    # 1. Pitch Detection (YIN)
    f0 = librosa.yin(y, fmin=60, fmax=400)
    f0 = f0[~np.isnan(f0)]

    if len(f0) > 0:
        avg_pitch = np.mean(f0)

        # NEW: Pitch Standard Deviation (Measures pitch instability)
        # A steady "Aaaaa" is stable (low value).
        # A singing voice or tremor is unstable (high value).
        pitch_std = np.std(f0)

        # We use this as our proxy for the "Tremor Instability Score"
        tremor_score = pitch_std
    else:
        avg_pitch = 0
        tremor_score = 0

    return y, sr, avg_pitch, tremor_score


# --- INPUT SECTION ---
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
    st.markdown("##### 🔍 Analyzing Pitch Stability...")

    try:
        y, sr, pitch, score = analyze_audio(audio_source)

        # --- RESULTS GRID ---
        col1, col2 = st.columns(2)
        col1.metric("Fundamental Freq", f"{pitch:.0f} Hz", delta_color="normal")
        col2.metric("Tremor Instability", f"{score:.2f}", "-1.0")

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

        # --- DIAGNOSIS LOGIC (Stability Based) ---
        st.divider()

        # LOGIC:
        # Monotone (Aaaa) -> Score usually below 5.0 -> GREEN
        # Unstable (Singing/Tremor) -> Score usually above 10.0 -> RED

        threshold = 8.0  # Threshold set to 8.0 to differentiate healthy/unstable

        if 50 < pitch < 400:
            if score > threshold:
                st.error("🚨 **Elevated Risk Detected**")
                st.markdown(
                    f"High Pitch Instability detected (Score: {score:.1f}). Recommendation: Clinical Screening."
                )
            else:
                st.success("✅ **Normal Biomarkers**")
                st.markdown(
                    f"Voice stability within healthy range (Score: {score:.1f})."
                )
        else:
            st.warning("⚠️ **Inconclusive**")
            st.markdown("Audio signal unclear or no voice detected.")

    except Exception as e:
        st.error(f"Analysis Error: {e}")
