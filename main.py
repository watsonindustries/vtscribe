"""
Microservice worker that uses OpenAI's Whisper model to download and transcribe videos.
"""
import modal
import pathlib
import dataclasses
import json
from typing import Iterator, TextIO, Tuple
from fastapi.responses import JSONResponse

app_image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg", "git")
    .pip_install(
        "ffmpeg-python",
        "torchaudio==0.12.1",
        "loguru==0.6.0",
        "git+https://github.com/yt-dlp/yt-dlp.git@master",
        "https://github.com/openai/whisper/archive/9f70a352f9f8630ab3aa0d06af5cb9532bd8c21d.tar.gz"
    )
)

stub = modal.Stub("vtscribe", image=app_image)

# Config


@dataclasses.dataclass
class ModelSpec:
    name: str
    params: str
    relative_speed: int  # Higher is faster


YT_DLP_DOWNLOAD_FILE_TEMPL = '%(id)s.%(ext)s'

DEFAULT_WHISPER_MODEL = ModelSpec(
    name="medium.en", params="769M", relative_speed=2)

CACHE_DIR = '/cache'
# Where downloaded VOD audio files are stored, by video ID.
RAW_AUDIO_DIR = pathlib.Path(CACHE_DIR, "raw_audio")
# Location of modal checkpoint.
MODEL_DIR = pathlib.Path(CACHE_DIR, "model")


volume = modal.SharedVolume().persist('vtscribe-cache-vol')

if stub.is_inside():
    from loguru import logger


@stub.webhook(method="POST", wait_for_response=True, timeout=40000)
def transcribe(video_id: str, model=DEFAULT_WHISPER_MODEL):
    download_audio.call('https://www.youtube.com/watch?v=' + video_id)
    input_raw_file: str = f"{CACHE_DIR}/{video_id}.mp3"
    result = do_transcribe.call(input_raw_file, model=model)

    return JSONResponse(content=result, status_code=200)


@stub.function(image=app_image, shared_volumes={CACHE_DIR: volume}, timeout=3000)
def download_audio(url: str):
    """Downloads audio track for a given web video URL as an mp3.

    Args:
        url (str): URL of a video to download audio track of.
    """
    import yt_dlp

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': CACHE_DIR + '/' + YT_DLP_DOWNLOAD_FILE_TEMPL
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        error_code = ydl.download(url)
        logger.info("Finished downloading with code: {}", error_code)


def format_timestamp(seconds: float, always_include_hours: bool = False, decimal_marker: str = '.') -> str:
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"


@stub.function(
    image=app_image,
    shared_volumes={CACHE_DIR: volume},
    cpu=2,
    timeout=3000
)
def transcribe_segment(
    start: float,
    end: float,
    audio_filepath: pathlib.Path,
    model: ModelSpec,
):
    import tempfile
    import time

    import ffmpeg
    import torch
    import whisper

    t0 = time.time()
    with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
        (
            ffmpeg.input(str(audio_filepath))
            .filter("atrim", start=start, end=end)
            .output(f.name)
            .overwrite_output()
            .run(quiet=True)
        )

        use_gpu = torch.cuda.is_available()
        device = "cuda" if use_gpu else "cpu"
        model = whisper.load_model(
            model.name, device=device, download_root=MODEL_DIR
        )
        result = model.transcribe(
            f.name, language="en", fp16=use_gpu)  # type: ignore

    logger.info(
        f"Transcribed segment {start:.2f} to {end:.2f} of {end - start:.2f} in {time.time() - t0:.2f} seconds."
    )

    # Add back offsets.
    for segment in result["segments"]:
        segment["start"] += start
        segment["end"] += start

    return result


@stub.function(image=app_image, shared_volumes={CACHE_DIR: volume}, timeout=30000)
def do_transcribe(audio_file_name: str, model: ModelSpec, result_path: str = "result.json") -> dict:
    """Perform transcription of an audio file using Whisper.
    Returns a Pandas dataframe

    Args:
        audio_file_name (str): path to mp3 audio file
    """
    import ffmpeg
    import time

    start_time = time.time()

    output_file = audio_file_name.replace(".mp3", ".wav")
    logger.info("Downsampling file {} --> {}", audio_file_name, output_file)
    stream = ffmpeg.input(audio_file_name)
    stream = ffmpeg.output(stream, output_file, **
                           {'ar': '16000', 'acodec': 'pcm_s16le'})
    ffmpeg.run(stream, overwrite_output=True, quiet=True)

    logger.info("Finished downsampling file: {}", output_file)
    logger.info("Preparing to transcribe file {}", output_file)

    segment_gen = split_silences(str(output_file))

    output_text = ""
    output_segments = []
    for result in transcribe_segment.starmap(
        segment_gen, kwargs=dict(audio_filepath=output_file, model=model)
    ):
        output_text += result["text"]
        output_segments += result["segments"]

    result = {
        "text": output_text,
        "segments": output_segments,
        "language": "en",
    }

    logger.info(f"Writing openai/whisper transcription to {result_path}")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=4)

    end_time = time.time()

    exec_time = end_time - start_time

    logger.info("Finished transcribing file {} in {} seconds",
                audio_file_name, exec_time)

    return result

# Adapted from https://github.com/modal-labs/modal-examples/blob/26c911ba880a1311e748c6b01f911d065aed4cc4/06_gpu_and_ml/whisper_pod_transcriber/pod_transcriber/main.py#L282


def split_silences(
    path: str, min_segment_length: float = 30.0, min_silence_length: float = 1.0
) -> Iterator[Tuple[float, float]]:
    """Split audio file into contiguous chunks using the ffmpeg `silencedetect` filter.
    Yields tuples (start, end) of each chunk in seconds."""

    import re
    import ffmpeg

    silence_end_re = re.compile(
        r" silence_end: (?P<end>[0-9]+(\.?[0-9]*)) \| silence_duration: (?P<dur>[0-9]+(\.?[0-9]*))"
    )

    metadata = ffmpeg.probe(path)
    duration = float(metadata["format"]["duration"])

    reader = (
        ffmpeg.input(str(path))
        .filter("silencedetect", n="-10dB", d=min_silence_length)
        .output("pipe:", format="null")
        .run_async(pipe_stderr=True)
    )

    cur_start = 0.0
    num_segments = 0

    while True:
        line = reader.stderr.readline().decode("utf-8")
        if not line:
            break
        match = silence_end_re.search(line)
        if match:
            silence_end, silence_dur = match.group("end"), match.group("dur")
            split_at = float(silence_end) - (float(silence_dur) / 2)

            if (split_at - cur_start) < min_segment_length:
                continue

            yield cur_start, split_at
            cur_start = split_at
            num_segments += 1

    # silencedetect can place the silence end *after* the end of the full audio segment.
    # Such segments definitions are negative length and invalid.
    if duration > cur_start and (duration - cur_start) > min_segment_length:
        yield cur_start, duration
        num_segments += 1
    print(f"Split {path} into {num_segments} segments")
