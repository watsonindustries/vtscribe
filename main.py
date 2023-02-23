"""
Microservice worker that uses OpenAI's Whisper model to download and transcribe videos.
"""
import modal
import pathlib
from typing import Iterator, TextIO
from fastapi.responses import JSONResponse

app_image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg", "git")
    .pip_install(
        "ffmpeg-python",
        "loguru==0.6.0",
        "yt-dlp",
        "git+https://github.com/m-bain/whisperx.git@main"
    )
)

stub = modal.Stub("vtscribe", image=app_image)

# Config

YT_DLP_DOWNLOAD_FILE_TEMPL = '%(id)s.%(ext)s'

WHISPER_MODEL_NAME = 'medium'

CACHE_DIR = '/cache'
# Where downloaded VOD audio files are stored, by video ID.
RAW_AUDIO_DIR = pathlib.Path(CACHE_DIR, "raw_audio")
# Location of modal checkpoint.
MODEL_DIR = pathlib.Path(CACHE_DIR, "model")

volume = modal.SharedVolume().persist('vtscribe-cache-vol')

if stub.is_inside():
    from loguru import logger


@stub.webhook(method="POST")
def transcribe(video_id: str):
    download_audio.call('https://www.youtube.com/watch?v=' + video_id)
    input_raw_file: str = f"{CACHE_DIR}/{video_id}.mp3"
    result = do_transcribe.call(input_raw_file)

    return JSONResponse(content=result, status_code=200)


@stub.function(image=app_image, shared_volumes={CACHE_DIR: volume})
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


def write_vtt(transcript: Iterator[dict], file: TextIO):
    print("WEBVTT\n", file=file)
    for segment in transcript:
        print(
            f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
            f"{segment['text'].strip().replace('-->', '->')}\n",
            file=file,
            flush=True,
        )


@stub.function(image=app_image, shared_volumes={CACHE_DIR: volume}, gpu="any")
def do_transcribe(audio_file_name: str) -> dict:
    """Perform transcription of an audio file using Whisper.
    Writes a VTT sub file to disk in the end in the format of `{audio_file_name}.vtt`
    Returns a Pandas dataframe

    Args:
        audio_file_name (str): path to mp3 audio file
    """
    import torch
    import whisperx
    import ffmpeg

    output_file = audio_file_name.replace(".mp3", ".wav")
    logger.info("Downsampling file {} --> {}", audio_file_name, output_file)
    stream = ffmpeg.input(audio_file_name)
    stream = ffmpeg.output(stream, output_file, **
                           {'ar': '16000', 'acodec': 'pcm_s16le'})
    ffmpeg.run(stream, overwrite_output=True, quiet=True)

    logger.info("Finished downsampling file: {}", output_file)
    logger.info("Preparing to transcribe file {}", output_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info("Using device: {} model: {}", device, WHISPER_MODEL_NAME)

    whisper_model = whisperx.load_model(
        WHISPER_MODEL_NAME, device=device, download_root=MODEL_DIR)
    result = whisper_model.transcribe(output_file)

    # Garbage collect everything before loading alignment model
    # del whisper_model
    # gc.collect()
    # torch.cuda.empty_cache()

    logger.info("Running alignment model...")

    # TODO: Figure out how to set cache root dir for align model download
    alignment_model, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device)

    result_aligned = whisperx.align(
        result["segments"], alignment_model, metadata, audio_file_name, device)

    res_segments = []

    for segment in result_aligned["segments"]:
        print(format_timestamp(segment['start']), segment['text'])
        res_segments.append(
            {'start': segment['start'], 'text': segment['text']})

    logger.info("Finished transcribing file {}", audio_file_name)

    return {'segments': res_segments, 'text': result['text'], 'language': result['language']}
