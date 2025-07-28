import torch
import torchaudio
import torchvision


class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)


class VideoTransform:
    def __init__(self, speed_rate):
        self.video_pipeline = torch.nn.Sequential(
            FunctionalModule(lambda x: x.unsqueeze(-1)),
            FunctionalModule(
                lambda x: (
                    x
                    if speed_rate == 1
                    else torch.index_select(
                        x,
                        dim=0,
                        index=torch.linspace(
                            0,
                            x.shape[0] - 1,
                            int(x.shape[0] / speed_rate),
                            dtype=torch.int64,
                        ),
                    )
                )
            ),
            FunctionalModule(lambda x: x.permute(3, 0, 1, 2)),
            FunctionalModule(lambda x: x / 255.0),
            torchvision.transforms.CenterCrop(88),
            torchvision.transforms.Normalize(0.421, 0.165),
        )

    def __call__(self, sample):
        return self.video_pipeline(sample)


class AudioTransform:
    def __init__(self):
        self.audio_pipeline = torch.nn.Sequential(
            FunctionalModule(
                lambda x: torch.nn.functional.layer_norm(x, x.shape, eps=0)
            ),
            FunctionalModule(lambda x: x.transpose(0, 1)),
        )

    def __call__(self, sample):
        return self.audio_pipeline(sample)


class AVSRDataLoader:
    def __init__(
        self,
        modality,
        speed_rate=1,
        transform=True,
        detector="mediapipe",
        convert_gray=True,
    ):
        self.modality = modality
        self.transform = transform
        if self.modality in ["video", "audiovisual"]:
            if detector == "mediapipe":
                from .video_process import VideoProcess

                self.video_process = VideoProcess(convert_gray=convert_gray)
            self.video_transform = VideoTransform(speed_rate=speed_rate)
        else:
            raise ValueError("Other detectors not currently supported")

    def load_data(self, data_filename, landmarks=None, transform=True):
        if self.modality == "video":
            video = self.load_video(data_filename)
            video = self.video_process(video, landmarks)
            video = torch.tensor(video)
            return self.video_transform(video) if self.transform else video
        if self.modality == "audiovisual":
            rate_ratio = 640
            audio, sample_rate = self.load_audio(data_filename)
            audio = self.audio_process(audio, sample_rate)
            video = self.load_video(data_filename)
            video = self.video_process(video, landmarks)
            video = torch.tensor(video)
            min_t = min(len(video), audio.size(1) // rate_ratio)
            audio = audio[:, : min_t * rate_ratio]
            video = video[:min_t]
            if self.transform:
                audio = self.audio_transform(audio)
                video = self.video_transform(video)
            return video, audio

    def load_audio(self, data_filename):
        waveform, sample_rate = torchaudio.load(data_filename, normalize=True)
        return waveform, sample_rate

    def load_video(self, data_filename):
        return torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()

    def audio_process(self, waveform, sample_rate, target_sample_rate=16000):
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, target_sample_rate
            )
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform
