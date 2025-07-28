import os

from google import genai
from google.genai import types

model_id = "gemini-2.5-flash"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

SAMPLE_VIDEO = "content/videos/preprocessed_sample.mp4"
SAMPLE_TRANSCRIPT = "WHAT I'M GOING TO DO IS I'M GOING TO HOLD MY HAND UP AND I'M GOING TO HOLD MY HAND UP AND I'M GOING TO HOLD MY HAND UP"


def summarise_video(video_path):
    """Generate summary of video with Gemini
    NB: video not immediately available after upload; need to load separately and then query them
    """
    if not GEMINI_API_KEY:
        raise ValueError("Please set your GEMINI_API_KEY")
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        video_bytes = open(video_path, "rb").read()

        prompt = f"""Try to infer what this scene represents. 
        Provide a brief summary of the environment, any individuals present and what they are doing. 
        Return only the summary"""

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
        print(f"Job Fetching Error: {e}")
        return None
    return response.text


if __name__ == "__main__":
    print(summarise_video(SAMPLE_VIDEO))
