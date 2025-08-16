import os

import aiofiles
from google import genai
from google.genai import types

model_id = "gemini-2.5-flash"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

async def summarise_video(video_path: str):
    """Generate summary of video clip with Gemini"""
    if not GEMINI_API_KEY:
        raise ValueError("Please set your GEMINI_API_KEY")
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        async with aiofiles.open(video_path, mode="rb") as f:
            video_bytes = await f.read()

        prompt = """
    Analyse this video segment and provide a concise, one-paragraph summary. 
    Focus on the main topic of conversation, key actions, or objects shown. 
    This summary will be used as context for a lip-reading AI.
    Return only the summary.
    """

        response = await client.aio.models.generate_content(
            model="models/gemini-2.0-flash",
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
        print(f"Summarise video error: {e}")
        return None
    return response.text

