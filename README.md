## AI‑Powered, Context‑Based Lip Reading

Generate transcripts for videos without audio using a visual speech recognition model (Auto‑AVSR) enhanced by contextual reasoning from Google Gemini. The system combines lip movements, visual context, and conversation history to improve transcript accuracy.

### How it works
1. Split the input video into ~15s segments.
2. For each segment (utilising concurrency with asyncio and parallelism with concurrent.futures where multiple GPUs available):
   - Preprocess to low‑res, low‑fps (optionally grayscale).
   - Generate a raw lip‑reading transcript locally with Auto‑AVSR.
   - Generate a concise visual summary using Gemini.
4. Send sequentially the raw transcript segments, the windowed context_history (i.e. the concatenated visual summaries) and corrected_transcript strings for previous segments to Gemini to produce final, corrected transcript segments.
5. Stitch segments with overlap handling to form a finished, corrected transcript for the entire video.
6. Having already updated the full preprocessed video to Gemini Files API, once it is ready on Google's side, feed the video and the final transcript to Gemini to diarise transcript.

### Prerequisites
- **Python**: 3.9–3.12
- **ffmpeg** (for MoviePy)
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt-get install ffmpeg`
- Optional: **GPU** with CUDA for faster inference (falls back to CPU)
- A Google Gemini API key: set `GEMINI_API_KEY`

### Installation
Using Poetry:
```bash
# Clone
git clone https://github.com/swhh/lip_reading_project.git
cd lip_reading_project

# Install deps
poetry install

# Install PyTorch stack manually (per project note)
poetry run pip install torch==2.2.2 torchaudio==2.2.2 torchvision==0.17.2
```

### Model weights
Place the Auto‑AVSR config and weights here:
- `src/lip_reading_project/content/model/LRS3_V_WER19.1.json`
- `src/lip_reading_project/content/model/LRS3_V_WER19.1_model.pth`

Use your own trained weights or obtain compatible ones from the upstream Auto‑AVSR project.

### Environment
Set your Gemini API key:
```bash
export GEMINI_API_KEY="YOUR_API_KEY"
```

### Data layout
- Put input videos in: `src/lip_reading_project/content/videos/`
- Segments will be written to: `src/lip_reading_project/content/videos/video_segments/`
- A preprocessed copy is created alongside the original with the prefix `preprocessed_`.

### Usage
From the project root:
```bash
# Basic
poetry run python src/lip_reading_project/main.py <filename.mp4>

# With segment overlap (seconds) to reduce word cuts across segments
poetry run python src/lip_reading_project/main.py <filename.mp4> --overlap 2
```
Notes:
- The CLI accepts `filename` (must exist in `content/videos/`),  `--overlap` and `--segment_len`.
- Internals use defaults: segment length ≈ 15s, preprocessing width 300px, fps 10, grayscale on.

### Output
- Transcript (pre‑ and post‑diarisation) is printed to stdout.
- If diarisation upload fails or is disabled, you’ll receive the corrected transcript without speaker labels.
- Redirect to a file if desired:
```bash
poetry run python src/lip_reading_project/main.py sample.mp4 --overlap 2 > transcript.txt
```

### Troubleshooting
- Missing API key: “Please set your GEMINI_API_KEY”
  - Ensure `export GEMINI_API_KEY="..."` is in your shell/env.
- Video not found: “Video file not found: content/videos/<file>”
  - Place the file in `src/lip_reading_project/content/videos/`.
- ffmpeg/MoviePy errors:
  - Install ffmpeg and re-run. On macOS: `brew install ffmpeg`.
  - Check (and change if necessary) codecs on line 46 of preprocess_video function in video_preprocessing file.
- Slow/CPU only:
  - Check CUDA availability; otherwise it will run on CPU, which will be really slow for the AVSR model.
- Model files not found:
  - Ensure the `.json` and `.pth` in `src/lip_reading_project/content/model/` match the expected names.



### Acknowledgements
- Auto‑AVSR: Lip‑reading Sentences Project. The pipelines module is a forked version of their GitHub project.
- ESPnet: End‑to‑End Speech Processing Toolkit






