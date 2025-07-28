import os

import cv2


SAMPLE_VIDEO = "content/videos/sample.mp4"


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


# --- Example Usage ---
if __name__ == "__main__":
    preprocess_video(
        SAMPLE_VIDEO, "content/videos/preprocessed_sample.mp4", 500, to_grayscale=True
    )
