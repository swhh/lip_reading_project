import argparse
import asyncio
from concurrent.futures import ProcessPoolExecutor
import math
import os

import torch

from utils import find_normalised_word_overlap, plausible_overlap
from final_transcript_generation import produce_transcript
from video_context_generation import summarise_video
from video_lipreading import InferencePipeline, modality, model_conf, model_path
from video_preprocessing import preprocess_video, split_video

VIDEO_DIR = "content/videos/"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AVG_WORDS_PER_SECOND = 1.5


def preprocess_wrapper(file_path):
    file_name = os.path.basename(file_path)
    output_path = file_path.replace(file_name, "preprocessed_" + file_name)
    preprocessed_video_path = preprocess_video(
        file_path,
        output_path=output_path,
        target_width=300,
        target_fps=10,
        to_grayscale=True,
    )
    return preprocessed_video_path


def run_inference_worker(
    segment_path: str, modality: str, model_path: str, model_conf: dict, device: str
) -> str:

    pipeline = InferencePipeline(
        modality=modality,
        model_path=model_path,
        model_conf=model_conf,
        face_track=True,
        device=device,
    )
    transcript = pipeline(segment_path)
    return transcript


async def main(filename, overlap=0):

    file_paths = split_video(os.path.join(VIDEO_DIR, filename), overlap=overlap)

    with ProcessPoolExecutor() as process_pool:
        loop = asyncio.get_running_loop()

        async def process_one_segment(segment_path: str):
            # Schedule the two CPU-bound tasks.
            transcript_future = loop.run_in_executor(
                process_pool,
                run_inference_worker,
                segment_path,
                modality,
                model_path,
                model_conf,
                DEVICE,
            )
            preprocessed_path_future = loop.run_in_executor(
                process_pool, preprocess_wrapper, segment_path
            )
            # Await the preprocessing result, as the summary task depends on it.
            preprocessed_video_path = await preprocessed_path_future

            summary_coroutine = summarise_video(preprocessed_video_path)
            # Await the remaining tasks.
            raw_transcript, summary = await asyncio.gather(
                transcript_future, summary_coroutine
            )

            # Return all three results.
            return raw_transcript, preprocessed_video_path, summary

        master_tasks = []
        for file_path in file_paths:
            master_tasks.append(process_one_segment(file_path))

        all_video_segment_info = await asyncio.gather(*master_tasks)

    full_transcript = (
        []
    )  # store transcript as a list to ensure word overlap calculations are efficient

    for (
        uncorrected_transcript,
        preprocessed_video_path,
        summary,
    ) in all_video_segment_info:

        corrected_transcript = produce_transcript(
            video_path=preprocessed_video_path,
            video_summary=summary,
            raw_transcript=uncorrected_transcript,
            conversation_history=full_transcript,
        )
        new_segment = corrected_transcript.split()
        if overlap:
            lookback_words = math.ceil(AVG_WORDS_PER_SECOND * overlap)
            full_transcript_lookback = " ".join(full_transcript[-lookback_words:])
            overlap_length = find_normalised_word_overlap(
                full_transcript_lookback, corrected_transcript
            )  # if overlap, might need to remove the overlap from the corrected transcript
            if plausible_overlap(
                overlap_length, overlap * AVG_WORDS_PER_SECOND, 0.4, 1.5
            ):  # check if overlap in right range to indicate duplicate content across segments
                new_segment = new_segment[overlap_length:]

        full_transcript.extend(new_segment)

    print(" ".join(full_transcript))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Lip-reading transcript generator",
        description="Generates a transcript of a video using a lip-reading AI model.",
    )
    parser.add_argument("filename")
    args = parser.parse_args()
    filename = args.filename

    asyncio.run(main(filename))
