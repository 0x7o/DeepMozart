# DeepMozart

Audio generation using diffusion models in PyTorch. The code is based on the [audio-diffusion-pytorch](https://github.com/archinetai/audio-diffusion-pytorch) repository.

## Installation

```bash
$ pip install DeepMozart
```

## Pretrained models

WIP

## Usage

```python
from DeepMozart import AudioDiffusionForUnconditionalGeneration
from audio_diffusion_pytorch import KarrasSchedule, AEulerSampler

model = AudioDiffForUnconditionalGeneration.from_pretrained("archinetai/audio-diffusion-pytorch/resolve/main/audio_1136_718k.pt")
sampler = AEulerSampler()
schedule = KarrasSchedule()

audio = model.generate(sampler, schedule, num_samples=1, steps=100, length=87)
audio[0].show()
```

