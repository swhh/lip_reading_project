# AI-Powered, Context-Based Lip Reading

## Summary

Project using the Google Gemini API and a forked version of the Auto-AVSR VSR model to lip-read people in videos based on, not only their lip movements, but their actions and the general video context.

##  How It Works
To get a transcript for a video, you run main.py with Poetry providing the file name for a video stored locally in the content/videos folder. The video is then split into segments of around 15 seconds. These segments are then pre-processed in parallel and the preprocessed versions sent to Gemini asynchronously to produce video summaries. At the same time, raw transcripts are generated for the unprocessed segments locally with the Auto-AVSR VSR model. Finally, the raw transcripts, the pre-processed videos and the video summaries are sent to Gemini synchronously to produce the final, corrected transcript for the entire video. 

