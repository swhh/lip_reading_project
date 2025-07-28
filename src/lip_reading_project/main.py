import torch

from final_transcript_generation import produce_transcript
from video_context_generation import summarise_video
from video_lipreading import InferencePipeline, modality, model_conf, model_path
from video_preprocessing import preprocess_video

VIDEO_PATH = "content/videos/new_sample.mp4"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    output_path = VIDEO_PATH.replace("new_sample", "preprocessed_new_sample")
    preprocessed_video_path = preprocess_video(
        VIDEO_PATH,
        output_path=output_path,
        target_width=300,
        target_fps=10,
        to_grayscale=True,
    )
    video_context = summarise_video(preprocessed_video_path)
    print(video_context)

    pipeline = InferencePipeline(
        modality, model_path, model_conf, face_track=True, device=DEVICE
    )
    uncorrected_transcript = pipeline(VIDEO_PATH)
    print("Uncorrected transcript:", uncorrected_transcript)

    corrected_transcript = produce_transcript(video_context, uncorrected_transcript)
    print("Corrected transcript:", corrected_transcript)

if __name__ == "__main__":
    main()