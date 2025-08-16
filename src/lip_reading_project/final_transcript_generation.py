import os

import asyncio
from google import genai
from google.genai import types

model_id = "gemini-2.5-flash"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


def _client():
    if not GEMINI_API_KEY:
        raise ValueError("Please set your GEMINI_API_KEY")
    return genai.Client(api_key=GEMINI_API_KEY)


def produce_transcript(
    video_path: str,
    video_summary: str,
    raw_transcript: str,
    conversation_history: str = "",
    context_history: str = ""
):
    """Produce final transcript segment with LLM based on uncorrected transcript from lip-reading model and AI-generated video context"""

    video_bytes = open(video_path, "rb").read()
    try:
        client = _client()

        prompt = f"""
                    You are an expert lip-reading assistant. Your task is to produce the definitive transcript for a video segment.

                    You have been provided with four sources of information:
                    1.  **Context from previous segments:** The corrected dialogue leading up to this point.
                    2.  **History from previous segments:** The high-level summaries of previous segments leading up to this point.
                    3.  **Video Summary:** A high-level summary of what happens in the current video segment.
                    4.  **Noisy Raw Transcript:** A raw, error-prone transcript from a specialized AI model.

                    **Context from previous segments:**
                    {conversation_history if conversation_history else "This is the first segment."}

                    **Context from previous segments:**
                    {context_history if context_history else "(none)"}

                    **Video Summary for CURRENT segment:**
                    "{video_summary}"

                    **Noisy Raw Transcript for CURRENT segment:**
                    "{raw_transcript}"

                    **Instructions:**
                    Synthesise all available information—the previous context, the summary, the noisy transcript, and MOST IMPORTANTLY, the visual evidence from the video itself—to produce the most accurate possible transcript for the current segment. 
                    Provide ONLY the corrected transcript.
                    """

        response = client.models.generate_content(
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
        print(f"Error producing transcript: {e}")
        return None
    return response.text



async def wait_until_file_active(file_name: str, timeout: int = 600, poll_interval: float = 3.0):
    client = _client()
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while True:
        file = await client.aio.files.get(name=file_name)
        state = getattr(file, "state", None)
        if state == "ACTIVE":
            return file
        if state in ("FAILED", "ERROR", "DELETED"):
            raise RuntimeError(f"File processing failed with state: {state}")
        if loop.time() >= deadline:
            raise TimeoutError("Timed out waiting for file to become ACTIVE")
        await asyncio.sleep(poll_interval)


async def upload_video_to_gemini(video_path: str):
    """
    Asynchronously upload a large video to Gemini Files API and return the uploaded file object.
    """
    client = _client()
    uploaded = await client.aio.files.upload(file=video_path)
    return uploaded 

def produce_global_diarised_transcript(
    video_file_uri: str,
    corrected_transcript: str,
    context_history: str = "",
) -> str:
    """
    Diarize the entire corrected transcript using the Files API file_uri for the full video.
    """
    try:
        client = _client()
        prompt = f"""
    You are an expert at speaker diarization using visual cues from the video and full conversational context.
    Attribute each utterance to stable speaker labels [S1], [S2], ... consistently across the entire video.

    Guidelines:
    - Use labels like [S1], [S2], etc. (no names).
    - Keep labels consistent throughout the whole transcript.
    - Preserve the words exactly; only add labels (minimal punctuation allowed).
    - Output format: lines like [S1]: <utterance>
    - Only add the next label when the speaker changes e.g. S1: some text newline S2: some other text

    Inputs:
    - Visual/context history (summaries from all segments): {context_history if context_history else "(none)"}
    - Full corrected transcript (no labels yet): "{corrected_transcript}"

    Return ONLY the diarized transcript, nothing else.
    """
        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=types.Content(
                parts=[
                    types.Part.from_uri(
                        file_uri=video_file_uri, mime_type="video/mp4"
                    ),
                    types.Part(text=prompt),
                ]
            ),
        )
    except Exception as e:
        print(e)
        return None
    
    return response.text


async def main():  
    uploaded_task = asyncio.create_task(upload_video_to_gemini('content/videos/video_segments/preprocessed_lip_read_sample_segment_001.mp4'))
    uploaded_file = await uploaded_task
    from time import sleep
    sleep(60)
    transcript = produce_global_diarised_transcript(uploaded_file.uri, 
                                       corrected_transcript="Hello and welcome to Charlie's lipreading test! So, are you a lipreading expert or a novice? This test is designed to help you find out! It's simple.",
                                       context_history="")
    print(transcript)



if __name__ == '__main__':
    #asyncio.run(main())
    file_path = "/Users/seamusholland/lip_reading_project/src/lip_reading_project/content/videos/preprocessed_new_sample.mp4"
    print(produce_transcript(file_path, "video of a woman talking", "completely uncontained environments"))
    
    



