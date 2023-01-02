from IPython.display import Audio
import torch
import math


class AudioDiffusionForUnconditionalGeneration():
    """
    Wrapper for the AudioDiffusion model for unconditional generation.
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device

    @classmethod
    def from_pretrained(cls, model_name_or_path, device="cpu", **kwargs):
        """
        model_name_or_path: str
        device: str
        """
        model = torch.hub.load_state_dict_from_url(
            model_name_or_path, map_location=device, **kwargs)
        return cls(model, device)

    def generate(self, sampler, schedule, num_samples=1, steps=100, length=12, sampling_rate=48000):
        """
        sampler: Sampler
        schedule: Schedule
        num_samples: int
        steps: int
        """
        length_samples = math.ceil(math.log2(length * sampling_rate))
        with torch.no_grad():
            samples = self.model.sample(
                noise=torch.randn(
                    (num_samples, 2, 2 ** length_samples), device=self.device),
                num_steps=steps,
                sigma_schedule=schedule,
                sampler=sampler,
            )
        return [Audio(x.cpu(), rate=sampling_rate) for x in samples]

