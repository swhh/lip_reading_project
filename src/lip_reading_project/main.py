import argparse
import asyncio
from concurrent.futures import ProcessPoolExecutor
import logging
import math
import os
import time

import torch
import torch.multiprocessing as mp


from final_transcript_generation import (
    produce_global_diarised_transcript,
    produce_transcript,
    upload_video_to_gemini,
    wait_until_file_active,
)
from utils import find_normalised_word_overlap, plausible_overlap
from video_context_generation import summarise_video
from video_lipreading import InferencePipeline, modality, model_conf, model_path
from video_preprocessing import preprocess_video, split_video


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
)
logger = logging.getLogger(__name__)

VIDEO_DIR = "content/videos/"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# set up multiprocessing so processpoolexecutor works with cuda
if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)

AVG_WORDS_PER_SECOND = 1.5
WINDOW_LENGTH = 2000


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


async def main(filename, overlap=0, window_length=WINDOW_LENGTH):
    start_time = time.time()

    try:
        if not filename:
            raise ValueError("Filename is required")

        original_video_path = os.path.join(VIDEO_DIR, filename)

        if not os.path.exists(original_video_path):
            raise FileNotFoundError(f"Video file not found: {original_video_path}")

        # 1. Preprocess full video once and use absolute path to ensure no issues with processpoolexecutor
        preprocessed_original_path = original_video_path.replace(
            filename, "preprocessed_" + filename
        )

        logger.info(f"Starting processing for video: {filename}")

        try:
            preprocess_video(
                original_video_path,
                output_path=preprocessed_original_path,
                target_width=300,
                target_fps=10,
                to_grayscale=True,
            )
            logger.info("Video preprocessing completed")
        except Exception as e:
            logger.error(f"Video preprocessing failed: {e}")
            raise RuntimeError(f"Cannot continue without preprocessed video: {e}")

        # 2. Start uploading the preprocessed full video early
        try:
            upload_task = asyncio.create_task(
                upload_video_to_gemini(preprocessed_original_path)
            )
        except Exception as e:
            logger.error(f"Failed to start video upload: {e}")
            upload_task = None

        file_paths = split_video(original_video_path, overlap=overlap)
        max_workers = (
            torch.cuda.device_count()
            if DEVICE == "cuda" and torch.cuda.device_count()
            else 1
        )
        # 3. generate segment context and raw transcripts
        with ProcessPoolExecutor(max_workers=max_workers) as process_pool:
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
                try:
                    preprocessed_video_path = await preprocessed_path_future
                    chosen_path = preprocessed_video_path
                except Exception as e:
                    logger.warning(f"Preprocessing failed for {segment_path}: {e}")
                    chosen_path = segment_path

                summary_coroutine = summarise_video(chosen_path)
                # Await the remaining tasks.
                raw_transcript, summary = await asyncio.gather(
                    transcript_future, summary_coroutine, return_exceptions=True
                )
                # Return all three results.
                return raw_transcript, chosen_path, summary

            master_tasks = []
            for file_path in file_paths:
                master_tasks.append(process_one_segment(file_path))

            segment_results = await asyncio.gather(
                *master_tasks, return_exceptions=True
            )

        all_video_segment_info = []
        failed_segments = []

        for i, result in enumerate(segment_results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process segment {i+1}: {result}")
                failed_segments.append((i + 1, str(result)))
                continue
            raw_transcript, chosen_path, summary = result

            if isinstance(raw_transcript, Exception):
                logger.warning(f"Failed to process segment {i+1}: {raw_transcript}")
                failed_segments.append((i + 1, str(raw_transcript)))
                continue
            
            if isinstance(summary, Exception):
                logger.warning(f"Summary for segment {i} failed: {summary}")
                summary = ""

            all_video_segment_info.append((raw_transcript, chosen_path, summary))
            logger.info(f"Successfully processed segment {i+1}/{len(file_paths)}")

        if not all_video_segment_info:
            raise RuntimeError("All video segments failed to process")

        if failed_segments:
            logger.warning(
                f"Processed {len(all_video_segment_info)} segments, {len(failed_segments)} failed"
            )

        
        # 4. generate full corrected transcript
        # store transcript as a list to ensure word overlap calculations are efficient
        corrected_transcript = []  
        context_history = ""

        for i, (
            uncorrected_transcript,
            preprocessed_video_path,
            summary,
        ) in enumerate(all_video_segment_info):
            
            try:
                corrected_segment = produce_transcript(
                    video_path=preprocessed_video_path,
                    video_summary=summary,
                    raw_transcript=uncorrected_transcript,
                    conversation_history=" ".join(corrected_transcript[-window_length:]),
                    context_history=context_history,
                )
            except Exception as e:
                logger.error(f'Failed to produce corrected segment for segment {i}: {e}', exc_info=True)
                corrected_segment = None
            
            if not corrected_segment:  # if no dialogue in segment
                continue

            new_segment = corrected_segment.split()
            if overlap:
                lookback_words = math.ceil(AVG_WORDS_PER_SECOND * overlap)
                full_transcript_lookback = " ".join(
                    corrected_transcript[-lookback_words:]
                )
                overlap_length = find_normalised_word_overlap(
                    full_transcript_lookback, corrected_segment
                )  # if overlap, might need to remove the overlap from the corrected transcript
                if plausible_overlap(
                    overlap_length, overlap * AVG_WORDS_PER_SECOND, 0.4, 1.5
                ):  # check if overlap in right range to indicate duplicate content across segments
                    new_segment = new_segment[overlap_length:]

            corrected_transcript.extend(new_segment)
            context_history += " " + summary

        if corrected_transcript:
            corrected_transcript = " ".join(corrected_transcript)
        print(
            "Transcript prior to diarisation:",
            corrected_transcript if corrected_transcript else "No transcript available",
        )

        #  5. Create final global diarised transcript with the file_uri
        uploaded_file = None
        if upload_task:
            try:
                uploaded_file = await upload_task
                logger.info("Video upload completed successfully")
            except Exception as e:
                logger.error(f"Video upload failed: {e}")

        if uploaded_file and getattr(uploaded_file, "state", None) != "ACTIVE":
            try:
                uploaded_file = await wait_until_file_active(
                    uploaded_file.name, timeout=600, poll_interval=10.0
                )
                logger.info("Video file is now active")
            except Exception as e:
                logger.error(f"Failed to wait for file activation: {e}")
                uploaded_file = None

        if uploaded_file:
            try:
                global_diarized = produce_global_diarised_transcript(
                    video_file_uri=uploaded_file.uri,
                    corrected_transcript=corrected_transcript,
                    context_history=context_history,
                )
                if global_diarized:
                    logger.info("Successfully generated diarised transcript")
                    print("Transcript with diarisation:", global_diarized)
                    return global_diarized
                else:
                    logger.warning("Diarisation failed, returning corrected transcript")
                    return corrected_transcript
            except Exception as e:
                logger.error(f"Diarisation failed: {e}")

        logger.info("Returning corrected transcript without diarisation")
        return corrected_transcript
    
    except Exception as e:
        logger.error(f"Critical error in main processing: {e}", exc_info=True)
        raise
    finally:
        processing_time = time.time() - start_time
        logger.info(f"Processing completed in {processing_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Lip-reading transcript generator",
        description="Generates a transcript of a video using a lip-reading AI model.",
    )
    parser.add_argument("filename")
    parser.add_argument("--overlap", type=int, default=0)
    args = parser.parse_args()
    filename = args.filename
    overlap = args.overlap

    asyncio.run(main(filename, overlap=overlap))
