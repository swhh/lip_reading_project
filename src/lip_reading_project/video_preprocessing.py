import os
import math

import cv2
from moviepy import VideoFileClip


def preprocess_video(
    input_path: str,
    output_path: str,
    target_width: int,
    target_fps: int = 10,
    to_grayscale: bool = False,
):
    """
    Preprocesses a video for AI, reducing FPS, resolution and
    optionally converting to grayscale.

    Args:
        input_path (str): Path to the source video.
        output_path (str): Path to save the processed video.
        target_width (int): The new width in pixels.
        target_fps (int): The new frames per second. (Default: 10)
        to_grayscale (bool): If True, convert the video to grayscale. (Default: False)
    """

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at: {input_path}")
    if os.path.exists(output_path):
        print(f"{output_path} already exists. No need to preprocess video at {input_path}.")
        return output_path

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Error opening video file: {input_path}")

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)

    aspect_ratio = original_height / original_width
    target_height = int(target_width * aspect_ratio)

    fourcc = cv2.VideoWriter_fourcc(*"avc1")

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        target_fps,
        (target_width, target_height),
        isColor=not to_grayscale,
    )

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if int(frame_count * target_fps / original_fps) > int(
            (frame_count - 1) * target_fps / original_fps
        ):
            resized_frame = cv2.resize(
                frame, (target_width, target_height), interpolation=cv2.INTER_AREA
            )

            # Convert to grayscale
            if to_grayscale:
                processed_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            else:
                processed_frame = resized_frame

            out.write(processed_frame)

        frame_count += 1

    cap.release()
    out.release()
    print(f"Successfully processed video and saved to: {output_path}")
    return output_path


def split_video(video_path, segment_length_sec=15, output_dir="content/videos/video_segments", overlap=0):
    """
    Splits a video into segments of a specified length.

    Args:
        video_path (str): Path to the input video file.
        segment_length_sec (int): The desired length of each segment in seconds.
        output_dir (str): The directory to save the video segments.
        overlap (int): The overlap in seconds between video segments to avoid transcription issues from cutting speaker off mid-word.

    Returns:
        list: A sorted list of file paths to the created video segments.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading video: {video_path}")
    clip = VideoFileClip(video_path)
    duration = clip.duration
    num_segments = math.ceil(duration / segment_length_sec)
    
    print(f"Video duration: {duration:.2f}s. Splitting into {num_segments} segments.")

    segment_paths = []
    for i in range(num_segments):
        start_time = i * segment_length_sec - overlap if i else 0
        end_time = min(start_time + segment_length_sec, duration)
        
        # Define the output filename
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_segment_{i+1:03d}.mp4")
        if os.path.exists(output_path): # check if already created
            print(F"Segment at {output_path} already exists.")
            segment_paths.append(output_path)
            continue
        
        print(f"  Creating segment {i+1}: from {start_time:.2f}s to {end_time:.2f}s -> {output_path}")
        
        # Create subclip and write to file
        subclip = clip.subclipped(start_time, end_time)
        subclip = subclip.without_audio()
        subclip.write_videofile(output_path, audio=False)
        
        segment_paths.append(output_path)
        subclip.close()

    clip.close()
    return sorted(segment_paths) 


