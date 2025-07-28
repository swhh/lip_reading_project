import os

from google import genai

model_id = "gemini-2.5-flash"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

SAMPLE_TRANSCRIPT = """WHAT I'M GOING TO DO IS I'M GOING TO HOLD MY HAND UP 
AND I'M GOING TO HOLD MY HAND UP AND I'M GOING TO HOLD MY HAND UP"""
SAMPLE_CONTEXT = """A happy toddler joyfully rides their small balance bike across an indoor space, 
showing off their newfound mobility. After a quick ride, they skillfully bring their toy to a stop and turn towards the camera. 
The child then proudly poses, beaming with a wide smile, seemingly delighted by their accomplishment."""


def produce_transcript(video_context: str, transcript: str):
    """Produce final transcript with LLM based on uncorrected transcript from lip-reading model and AI-generated video context"""
    if not GEMINI_API_KEY:
        raise ValueError("Please set your GEMINI_API_KEY")
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        prompt = f""" 
        Context: {video_context}
        Transcript: {transcript}.
        Correct the transcript based on the provided context.
        Return only the corrected transcript."""

        response = client.models.generate_content(model=model_id, contents=prompt)
    except Exception as e:
        print(f"Job Fetching Error: {e}")
        return None
    return response.text


if __name__ == "__main__":
    print(produce_transcript(SAMPLE_CONTEXT, SAMPLE_TRANSCRIPT))
