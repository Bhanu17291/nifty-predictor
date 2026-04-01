import os
import streamlit as st

def generate_voice_briefing(commentary: str) -> bytes | None:
    try:
        from gtts import gTTS
        import io
        tts = gTTS(text=commentary, lang="en", slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except ImportError:
        return None

def render_voice_briefing(commentary: str):
    st.markdown('<p class="sec-label">Voice Briefing</p>', unsafe_allow_html=True)
    audio_bytes = generate_voice_briefing(commentary)
    if audio_bytes:
        st.audio(audio_bytes, format="audio/mp3")
    else:
        st.info("Install gTTS to enable voice: `pip install gTTS`")