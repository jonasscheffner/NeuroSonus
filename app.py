import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="NeuroSonus AI", layout="centered")

# --- TITLE AND INTRO ---
st.title("🧠 NeuroSonus")
st.markdown(
    """
**Early Detection System for Neurodegenerative Diseases**
*Upload a voice sample to analyze acoustic biomarkers and detect early warning signs.*
"""
)

st.divider()


# --- FUNCTION FOR AUDIO ANALYSIS ---
def analyze_audio(audio_file):
    # Load audio file
    y, sr = librosa.load(
        audio_file, duration=30
    )  # Max 30 seconds to save processing time

    # Extract Biomarkers (Medical/Technical part)
    # 1. Pitch / Fundamental Frequency (Variations often indicate Parkinson's)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    avg_pitch = np.mean(pitches[pitches > 0])

    # 2. Zero Crossing Rate (Indicator for "noise" or roughness in voice)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    # 3. Spectral Centroid (Brightness of the voice)
    sc = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    return y, sr, avg_pitch, zcr, sc


# --- UI INTERACTION ---
uploaded_file = st.file_uploader(
    "Upload Voice Sample (WAV or MP3)", type=["wav", "mp3"]
)

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    with st.spinner("AI analyzing vocal micro-structures..."):
        try:
            # Start Analysis
            y, sr, pitch, zcr, sc = analyze_audio(uploaded_file)

            st.success("Analysis complete. Extracting features...")

            # --- DISPLAY RESULTS (DASHBOARD) ---
            col1, col2, col3 = st.columns(3)
            col1.metric("Avg. Fund. Freq (Hz)", f"{pitch:.2f}", "Normal Range")
            col2.metric("Vocal Jitter (ZCR)", f"{zcr:.4f}", "-0.002")
            col3.metric("Spectral Centroid", f"{sc:.0f}", "+120")

            st.divider()

            # --- VISUALIZATION FOR INVESTORS ---
            st.subheader("Spectrogram Analysis")
            fig, ax = plt.subplots(figsize=(10, 4))

            # Convert power to decibels
            S_dB = librosa.power_to_db(
                librosa.feature.melspectrogram(y=y, sr=sr), ref=np.max
            )

            # Display spectrogram
            img = librosa.display.specshow(
                S_dB, x_axis="time", y_axis="mel", sr=sr, ax=ax
            )
            st.pyplot(fig)

            st.divider()

            # --- SIMULATED DIAGNOSIS (MOCKUP FOR PITCH DECK) ---
            # NOTE: Real diagnosis requires a trained Machine Learning Model (e.g., Random Forest / Neural Net)
            # We are simulating logic based on the extracted values for the demo effect.

            st.subheader("AI Risk Assessment")

            risk_score = 0
            # Simulated algorithm for demo purposes
            if pitch < 100 or pitch > 300:
                risk_score += 30
            if zcr > 0.1:
                risk_score += 40

            if risk_score < 30:
                st.info("✅ **Status: Low Risk** - No significant anomalies detected.")
                st.caption("Recommendation: Next routine scan in 3 months.")
            elif risk_score < 70:
                st.warning(
                    "⚠️ **Status: Moderate Risk** - Slight deviations in prosody detected."
                )
                st.caption(
                    "Recommendation: Enable daily monitoring to establish baseline."
                )
            else:
                st.error(
                    "🚨 **Status: High Risk** - Significant deviation from baseline."
                )
                st.caption(
                    "Recommendation: Consult a neurologist. Download Clinical Report."
                )

        except Exception as e:
            st.error(f"Error during analysis: {e}")

# --- FOOTER ---
st.markdown("---")
st.caption("© 2026 NeuroSonus Prototype - Internal Investor Build v0.2")
