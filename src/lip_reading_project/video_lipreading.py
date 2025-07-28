import os

import torch

from pipelines.model import AVSR
from pipelines.data import AVSRDataLoader
from pipelines.detector import LandmarksDetector


MODEL_ID = "LRS3_V_WER19.1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 3. Path to your preprocessed data or video
VIDEO_PATH = "content/videos/sample.mp4"

modality = "video"
model_conf = f"content/model/{MODEL_ID}.json"
model_path = f"content/model/{MODEL_ID}_model.pth"


class InferencePipeline(torch.nn.Module):
    def __init__(
        self,
        modality,
        model_path,
        model_conf,
        detector="mediapipe",
        face_track=False,
        device="cuda:0",
    ):
        super(InferencePipeline, self).__init__()
        self.device = device
        # modality configuration
        self.modality = modality
        self.dataloader = AVSRDataLoader(modality, detector=detector)
        self.model = AVSR(
            modality,
            model_path,
            model_conf,
            rnnlm=None,
            rnnlm_conf=None,
            penalty=0.0,
            ctc_weight=0.1,
            lm_weight=0.0,
            beam_size=40,
            device=device,
        )
        if face_track and self.modality in ["video", "audiovisual"]:
            self.landmarks_detector = LandmarksDetector()
        else:
            self.landmarks_detector = None

    def process_landmarks(self, data_filename, landmarks_filename):
        if self.modality == "audio":
            return None
        if self.modality in ["video", "audiovisual"]:
            landmarks = self.landmarks_detector(data_filename)
            return landmarks

    def forward(self, data_filename, landmarks_filename=None):
        assert os.path.isfile(
            data_filename
        ), f"data_filename: {data_filename} does not exist."
        landmarks = self.process_landmarks(data_filename, landmarks_filename)
        data = self.dataloader.load_data(data_filename, landmarks)
        transcript = self.model.infer(data)
        return transcript

    def extract_features(
        self, data_filename, landmarks_filename=None, extract_resnet_feats=False
    ):
        assert os.path.isfile(
            data_filename
        ), f"data_filename: {data_filename} does not exist."
        landmarks = self.process_landmarks(data_filename, landmarks_filename)
        data = self.dataloader.load_data(data_filename, landmarks)
        with torch.no_grad():
            if isinstance(data, tuple):
                enc_feats = self.model.model.encode(
                    data[0].to(self.device),
                    data[1].to(self.device),
                    extract_resnet_feats,
                )
            else:
                enc_feats = self.model.model.encode(
                    data.to(self.device), extract_resnet_feats
                )
        return enc_feats


def run_inference(video_path: str, pipeline: InferencePipeline):
    """Runs the full lip-reading pipeline on a video."""
    return pipeline(video_path)


if __name__ == "__main__":
    print("Setting up pipeline\n")
    pipeline = InferencePipeline(
        modality, model_path, model_conf, face_track=True, device=DEVICE
    )
    print("Running inference\n")
    transcript = pipeline(VIDEO_PATH)
    print(transcript)
