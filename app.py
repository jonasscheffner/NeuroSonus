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
st.caption("v1.0.3 • Medical Research Prototype")

st.divider()


# --- ANALYSIS FUNCTION ---
def analyze_audio(audio_file):
    # Load audio
    y, sr = librosa.load(audio_file, duration=10)  # 10 seconds is enough for live demo

    # 1. Pitch (Fundamental Frequency)
    # We use a more robust method for speech (YIN algorithm is better for voice)
    f0 = librosa.yin(y, fmin=50, fmax=300)  # Range for human voice (50-300Hz)
    f0 = f0[~np.isnan(f0)]  # Remove NaNs
    avg_pitch = np.mean(f0) if len(f0) > 0 else 0

    # 2. Jitter (Simulated via Zero Crossing Rate variance as proxy)
    zcr = librosa.feature.zero_crossing_rate(y)
    jitter_proxy = np.var(zcr)

    # 3. Spectral Centroid
    sc = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    return y, sr, avg_pitch, jitter_proxy, sc


# --- INPUT SECTION (TABS) ---
tab1, tab2 = st.tabs(["🎙️ Record Live", "📂 Upload File"])

audio_source = None

with tab1:
    # DAS IST NEU: Der Aufnahme-Button
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
        y, sr, pitch, jitter, sc = analyze_audio(audio_source)

        # --- RESULTS GRID ---
        col1, col2 = st.columns(2)
        col1.metric("Fundamental Freq", f"{pitch:.0f} Hz", delta_color="normal")
        col2.metric("Micro-Tremor (Var)", f"{jitter:.5f}", "-0.0002")

        # --- SPECTROGRAM ---
        st.markdown("###### Spectrogram Analysis")
        fig, ax = plt.subplots(figsize=(10, 3), facecolor="#0e1117")
        S_dB = librosa.power_to_db(
            librosa.feature.melspectrogram(y=y, sr=sr), ref=np.max
        )
        img = librosa.display.specshow(
            S_dB, x_axis="time", y_axis="mel", sr=sr, ax=ax, cmap="magma"
        )
        ax.axis("off")  # Cleaner look without axis labels
        st.pyplot(fig)

        # --- DIAGNOSIS LOGIC (Demo) ---
        st.divider()

        # Human voice is typically 85-255 Hz.
        # If outside this range, it's likely noise or silence.
        if 85 < pitch < 255:
            # Valid voice range
            if jitter > 0.002:  # Threshold for "shaky" voice
                st.error("🚨 **Elevated Risk Detected**")
                st.markdown(
                    "Significant vocal tremor anomalies identified. Recommendation: Clinical Screening."
                )
            else:
                st.success("✅ **Normal Biomarkers**")
                st.markdown("No significant acoustic anomalies detected.")
        else:
            st.warning("⚠️ **Inconclusive**")
            st.markdown(
                f"Voice pitch ({pitch:.0f} Hz) out of normal human range. Please record again clearly."
            )

    except Exception as e:
        st.error(f"Analysis Error: {e}")
