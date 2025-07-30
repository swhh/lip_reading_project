import os

from google import genai
from google.genai import types

model_id = "gemini-2.5-flash"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

def produce_transcript(video_path: str, video_summary: str, raw_transcript: str, conversation_history:str = ""):
    """Produce final transcript with LLM based on uncorrected transcript from lip-reading model and AI-generated video context"""
    if not GEMINI_API_KEY:
        raise ValueError("Please set your GEMINI_API_KEY")
    
    video_bytes = open(video_path, "rb").read()
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        prompt = f"""
                    You are an expert lip-reading assistant. Your task is to produce the definitive transcript for a video segment.

                    You have been provided with three sources of information:
                    1.  **Context from previous segments:** The corrected dialogue leading up to this point.
                    2.  **Video Summary:** A high-level summary of what happens in the current video segment.
                    3.  **Noisy Raw Transcript:** A raw, error-prone transcript from a specialized AI model.

                    **Context from previous segments:**
                    {conversation_history if conversation_history else "This is the first segment."}

                    **Video Summary for CURRENT segment:**
                    "{video_summary}"

                    **Noisy Raw Transcript for CURRENT segment:**
                    "{raw_transcript}"

                    **Instructions:**
                    Synthesise all available information—the previous context, the summary, the noisy transcript, and MOST IMPORTANTLY, the visual evidence from the video itself—to produce the most accurate possible transcript for the current segment. 
                    Provide ONLY the corrected transcript.
                    """


        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=types.Content(
                parts=[
                    types.Part(
                        inline_data=types.Blob(data=video_bytes, mime_type="video/mp4")
                    ),
                    types.Part(text=prompt),
                ]
            ),
        )
    except Exception as e:
        print(f"Error producing transcript: {e}")
        return None
    return response.text


