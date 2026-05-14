#!/usr/bin/env python3
"""Wyoming STT server for IBM Granite Speech 4.1 2B.

Adapted from the Wyoming-Granite-STT reference implementation by mikey60:
https://github.com/mikey60/Wyoming-Granite-STT

Changes from the original Granite 4.0 1B implementation:
  - Target model: granite-speech-4.1-2b (2B params, bfloat16 native)
  - Updated prompt format for Granite 4.1 chat template
  - Added punctuation/capitalization support (native in 4.1)
  - Added Japanese language support
  - Uses Hugging Face device_map for auto device placement
  - Default dtype: bfloat16 (recommended for Granite 4.1)
"""

import argparse
import asyncio
import logging
import os
import tempfile
import wave
from functools import partial
from pathlib import Path
from typing import Optional

import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

_LOGGER = logging.getLogger("wyoming-granite-stt")

LANG_NAME: dict[str, str] = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "ja": "Japanese",
}

LANG_CODES = sorted(LANG_NAME.keys())


def norm_lang(lang: Optional[str]) -> Optional[str]:
    if not lang:
        return None
    return lang.split("-")[0].lower()


class GraniteTranscriber:
    def __init__(
        self,
        model_id: str,
        device: str,
        dtype: str,
        max_new_tokens: int,
        num_beams: int,
        punctuation: bool,
    ):
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.punctuation = punctuation

        if dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

        _LOGGER.info("Loading processor: %s", model_id)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer

        _LOGGER.info(
            "Loading model: %s (device=%s dtype=%s)", model_id, device, dtype
        )
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            dtype=torch_dtype,
            device_map=device,
        )
        self.model.eval()

        self._lock = asyncio.Lock()

    def _transcribe_sync(self, wav_path: str, language: Optional[str]) -> str:
        wav, sr = torchaudio.load(wav_path, normalize=True)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)

        # Build prompt using the Granite Speech 4.1 prompt style
        if self.punctuation:
            prompt_text = (
                "<|audio|>transcribe the speech "
                "with proper punctuation and capitalization."
            )
        else:
            lang_key = norm_lang(language)
            if lang_key and lang_key in LANG_NAME:
                prompt_text = (
                    f"<|audio|>can you transcribe the speech "
                    f"into a written format? "
                    f"The spoken language is {LANG_NAME[lang_key]}."
                )
            else:
                prompt_text = (
                    "<|audio|>can you transcribe the speech "
                    "into a written format?"
                )

        chat = [{"role": "user", "content": prompt_text}]
        prompt = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

        model_inputs = self.processor(
            prompt, wav, device=self.device, return_tensors="pt"
        )

        with torch.inference_mode():
            out = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=self.num_beams,
            )

        num_in = model_inputs["input_ids"].shape[-1]
        gen = out[:, num_in:]
        text = self.tokenizer.batch_decode(
            gen, add_special_tokens=False, skip_special_tokens=True
        )[0].strip()
        return text

    async def transcribe(self, wav_path: str, language: Optional[str]) -> str:
        async with self._lock:
            return await asyncio.to_thread(
                self._transcribe_sync, wav_path, language
            )


class GraniteEventHandler(AsyncEventHandler):
    def __init__(
        self,
        wyoming_info: Info,
        transcriber: GraniteTranscriber,
        default_language: Optional[str],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.wyoming_info_event = wyoming_info.event()
        self.transcriber = transcriber
        self.default_language = default_language

        self._language: Optional[str] = None
        self._wav_dir = tempfile.TemporaryDirectory()
        self._wav_path = os.path.join(self._wav_dir.name, "speech.wav")
        self._wav_file: Optional[wave.Wave_write] = None

        self._audio_converter = AudioChunkConverter(
            rate=16000, width=2, channels=1
        )

    async def handle_event(self, event: Event) -> bool:
        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            chunk = self._audio_converter.convert(chunk)

            if self._wav_file is None:
                self._wav_file = wave.open(self._wav_path, "wb")
                self._wav_file.setframerate(chunk.rate)
                self._wav_file.setsampwidth(chunk.width)
                self._wav_file.setnchannels(chunk.channels)

            self._wav_file.writeframes(chunk.audio)
            return True

        if Transcribe.is_type(event.type):
            t = Transcribe.from_event(event)
            self._language = t.language or self.default_language
            return True

        if AudioStop.is_type(event.type):
            if self._wav_file is not None:
                self._wav_file.close()
                self._wav_file = None

            lang = self._language
            text = await self.transcriber.transcribe(self._wav_path, lang)
            _LOGGER.info("Transcript (%s): %s", lang, text)

            await self.write_event(
                Transcript(text=text, language=norm_lang(lang)).event()
            )

            self._language = None
            return False

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            return True

        return True


async def main() -> None:
    ap = argparse.ArgumentParser(
        description="Wyoming STT server for IBM Granite Speech 4.1 2B",
    )
    ap.add_argument(
        "--uri", default="tcp://0.0.0.0:10300", help="Server URI to listen on"
    )
    ap.add_argument(
        "--model",
        default="ibm-granite/granite-speech-4.1-2b",
        help="Hugging Face model ID",
    )
    ap.add_argument(
        "--device", default="cuda", help="Device: cuda, cpu, auto"
    )
    ap.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
        help="Model precision (bfloat16 recommended for Granite 4.1)",
    )
    ap.add_argument(
        "--language",
        default="en-US",
        help="Default language for Home Assistant (e.g. en-US)",
    )
    ap.add_argument(
        "--max-new-tokens", type=int, default=256, help="Max generated tokens"
    )
    ap.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="Beam search width (1 = greedy, 2+ = more accurate but slower)",
    )
    ap.add_argument(
        "--punctuation",
        action="store_true",
        default=True,
        help="Enable punctuation and capitalization (default: on)",
    )
    ap.add_argument(
        "--no-punctuation",
        action="store_false",
        dest="punctuation",
        help="Disable punctuation and capitalization",
    )
    ap.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="granite-speech-stt",
                description="IBM Granite Speech 4.1 2B (ASR) via Transformers",
                attribution=Attribution(
                    name="IBM", url="https://huggingface.co/ibm-granite"
                ),
                installed=True,
                version="0.1.0",
                models=[
                    AsrModel(
                        name=args.model,
                        description=f"Granite Speech 4.1 2B — multilingual ASR "
                        f"(en/fr/de/es/pt/ja) with punctuation",
                        attribution=Attribution(
                            name="IBM",
                            url="https://huggingface.co/ibm-granite",
                        ),
                        installed=True,
                        languages=LANG_CODES,
                        version="4.1-2b",
                    )
                ],
            )
        ]
    )

    transcriber = GraniteTranscriber(
        model_id=args.model,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        punctuation=args.punctuation,
    )

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Ready on %s", args.uri)
    await server.run(
        partial(
            GraniteEventHandler,
            wyoming_info,
            transcriber,
            args.language,
        )
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
