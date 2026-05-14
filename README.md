# Wyoming Granite STT

Wyoming protocol STT server for [IBM Granite Speech 4.1 2B](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) — a multilingual speech-to-text model supporting English, French, German, Spanish, Portuguese, and Japanese, with punctuation and capitalization.

Adapted from [mikey60/Wyoming-Granite-STT](https://github.com/mikey60/Wyoming-Granite-STT).

## Features

- **6 languages**: EN, FR, DE, ES, PT, JA
- **Punctuation & capitalization** enabled by default
- **bfloat16** — lower VRAM (~4–5 GB), runs comfortably on modest GPUs
- **Wyoming protocol** — drop-in STT for [Home Assistant](https://www.home-assistant.io/integrations/wyoming)
- **Docker** with GPU passthrough or bare-metal with `uv venv`

## Quick start

### Docker (recommended)

```bash
git clone https://github.com/v0l/wyoming-granite-stt.git
cd wyoming-granite-stt
docker compose build
docker compose up -d
```

### Bare metal

```bash
git clone https://github.com/v0l/wyoming-granite-stt.git
cd wyoming-granite-stt
uv venv
uv pip install -r requirements.txt
./run.sh
```

## Configuration

| Flag               | Default                             | Description                               |
| ------------------ | ----------------------------------- | ----------------------------------------- |
| `--uri`            | `tcp://0.0.0.0:10300`               | Server listen address                     |
| `--model`          | `ibm-granite/granite-speech-4.1-2b` | Hugging Face model ID                     |
| `--device`         | `cuda`                              | `cuda`, `cpu`, or `auto`                  |
| `--dtype`          | `bfloat16`                          | `float16`, `bfloat16`, `float32`          |
| `--language`       | `en-US`                             | Default language for HA                   |
| `--num-beams`      | `1`                                 | Beam width (1 = fast, 2+ = more accurate) |
| `--max-new-tokens` | `256`                               | Max output tokens                         |
| `--no-punctuation` | —                                   | Disable punctuation/capitalization        |
| `--debug`          | —                                   | Verbose logging                           |

## Home Assistant

Add the **Wyoming Protocol** integration and point it at `tcp://<host>:10300`. The server advertises itself as `granite-speech-stt`.

## Performance

Roughly 0.5–1.0 seconds per utterance with `--num-beams 1` on a modern NVIDIA GPU. The 2B model fits in ~4–5 GB VRAM at bfloat16.

## Credits

- [IBM Granite Speech Team](https://huggingface.co/ibm-granite) — model
- [mikey60/Wyoming-Granite-STT](https://github.com/mikey60/Wyoming-Granite-STT) — original Wyoming wrapper for Granite 4.0
- [OHF-Voice/wyoming](https://github.com/OHF-Voice/wyoming) — Wyoming protocol library
