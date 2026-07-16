import streamlit as st
import os
import sys
import tempfile

# --- FIX: Ensure Python can find the 'core' directory on Streamlit Cloud ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.models import transcribe_audio, calculate_semantic_similarity
from core.audio import analyze_audio_fluency

# Page Configuration
st.set_page_config(
    page_title="VBCUA - Concept Analyser",
    page_icon="🎙️",
    layout="wide"
)

# Application Header
st.title("🎙️ Voice-Based Concept Understanding Analyser (VBCUA)")
st.markdown("---")

# Layout Configuration: Sidebar for references, Main body for processing
with st.sidebar:
    st.header("🎯 Reference Configurations")
    
    # Pre-populate with a standard technical concept definition as an example
    default_concept = (
        "Machine learning is a branch of artificial intelligence focused on building applications "
        "that learn from data and improve their accuracy over time without being explicitly programmed."
    )
    
    reference_concept = st.text_area(
        "Gold-Standard Concept Definition:",
        value=default_concept,
        height=200,
        help="Enter the exact core definition you want the speaker to match against."
    )
    
    st.markdown("---")
    st.caption("Developed by the VBCUA Project Team.")

# Main Application Dashboard
st.subheader("1. Submit Your Concept Audio Explanation")
uploaded_file = st.file_uploader(
    "Upload audio format (.wav or .mp3)", 
    type=["wav", "mp3"],
    help="Record your concept explanation using a voice recorder tool and upload the file here."
)

if uploaded_file is not None:
    # --- FIX 1: Write to system temporary folder to prevent infinite restart loops ---
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_filename = temp_file.name
        
    # Provide audio playback for verification
    st.audio(temp_filename)
    
    st.markdown("### ⚡ Analytics Processing Control")
    if st.button("Run AI Assessment Engine", type="primary"):
        if not reference_concept.strip():
            st.warning("Please provide a target reference definition in the sidebar before evaluation.")
        else:
            # Create a loading spinner while computation is running
            with st.spinner("AI is evaluating your audio file... This involves downloading weights if it's the first execution."):
                
                # Execution Pipeline Block A: Speech-to-Text
                transcript = transcribe_audio(temp_filename)
                
                # Execution Pipeline Block B: Semantic Similarity Matching
                similarity_score = calculate_semantic_similarity(transcript, reference_concept)
                
                # Execution Pipeline Block C: Audio Signal Analysis
                fluency = analyze_audio_fluency(temp_filename, transcript)
                
            # --- RENDER RESULTS VISUALLY ---
            st.success("Analysis Complete!")
            st.markdown("---")
            
            st.subheader("2. Speech Evaluation Output Metrics")
            
            # Metric Callout Cards
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    label="Conceptual Accuracy", 
                    value=f"{round(similarity_score * 100, 1)}%"
                )
            with col2:
                st.metric(
                    label="Speaking Duration", 
                    value=f"{fluency['duration_seconds']} s"
                )
            with col3:
                st.metric(
                    label="Silence / Pause Ratio", 
                    value=f"{int(fluency['pause_ratio'] * 100)}%"
                )
            with col4:
                st.metric(
                    label="Filler Words Detected", 
                    value=fluency['filler_words_detected']
                )
                
            # Content Breakdowns
            st.markdown("### 📝 Spoken Text Transcription")
            if transcript:
                st.info(transcript)
            else:
                st.error("Could not confidently convert speech to text. Ensure the audio is clear and audible.")
                
            st.markdown("### 🎯 Expected Concept Definition")
            st.code(reference_concept, language="markdown")
            
            # --- FIX 2: Only delete the temp file AFTER processing is fully done ---
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
else:
    st.info("Waiting for an audio file submission to initialize evaluation pipelines.")
