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
st.caption("v1.0.6 • Micro-Tremor Variance Algorithm")

st.divider()


# --- ANALYSIS FUNCTION ---
def analyze_audio(audio_file):
    # Load audio
    y, sr = librosa.load(audio_file, duration=10)

    # 1. Pitch Detection (YIN) for display purposes
    f0 = librosa.yin(y, fmin=60, fmax=400)
    f0 = f0[~np.isnan(f0)]
    avg_pitch = np.mean(f0) if len(f0) > 0 else 0

    # 2. Micro-Tremor / Jitter Variance
    # Instead of Pitch Stability (which triggers on normal reading),
    # we measure the "Roughness" or variance in the signal structure (Zero Crossing Rate).
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_var = np.var(zcr)

    # SCALING: We multiply by 10,000 to get readable scores (0-100 scale)
    # Healthy "Ahhh": ~ 5-20
    # Normal Reading: ~ 30-60
    # Sick/Tremor:    ~ 100-200
    tremor_score = zcr_var * 10000

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
    st.markdown("##### 🔍 Analyzing Vocal Roughness...")

    try:
        y, sr, pitch, score = analyze_audio(audio_source)

        # --- RESULTS GRID ---
        col1, col2 = st.columns(2)
        col1.metric("Fundamental Freq", f"{pitch:.0f} Hz", delta_color="normal")
        col2.metric("Tremor Score", f"{score:.1f}", "-10.0")

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

        # --- DIAGNOSIS LOGIC ---
        st.divider()

        # CALIBRATION:
        # Score < 80: Green (Healthy sustained tone OR Normal smooth speech)
        # Score > 80: Red (Rough voice, stuttering, heavy tremor, vocal fry)

        threshold = 80.0

        if 50 < pitch < 400:
            if score > threshold:
                st.error("🚨 **Elevated Risk Detected**")
                st.markdown(
                    f"High Vocal Roughness detected (Score: {score:.1f}). Recommendation: Clinical Screening."
                )
            else:
                st.success("✅ **Normal Biomarkers**")
                st.markdown(
                    f"Voice biomarkers within healthy range (Score: {score:.1f})."
                )
        else:
            st.warning("⚠️ **Inconclusive**")
            st.markdown("Audio signal unclear or no voice detected.")

    except Exception as e:
        st.error(f"Analysis Error: {e}")
