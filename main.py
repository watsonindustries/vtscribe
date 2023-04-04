"""
Microservice worker that uses OpenAI's Whisper model to download and transcribe videos.
"""
import modal
import pathlib
import dataclasses
import json
from typing import Iterator, Tuple
from fastapi.responses import JSONResponse

app_image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg", "git")
    .pip_install(
        "boto3",
        "ffmpeg-python",
        "torchaudio==0.12.1",
        "loguru==0.6.0",
        "git+https://github.com/yt-dlp/yt-dlp.git@master",
        "https://github.com/openai/whisper/archive/9f70a352f9f8630ab3aa0d06af5cb9532bd8c21d.tar.gz"
    )
)

MODAL_STUB_NAME = 'vtscribe'
MODAL_SECRETS_NAME = 'vtscribe-secrets'

DO_DEFAULT_BUCKET_NAME = 'holosays'

stub = modal.Stub(MODAL_STUB_NAME, image=app_image, secrets=[
                  modal.Secret.from_name(MODAL_SECRETS_NAME)])

# Config


@dataclasses.dataclass
class ModelSpec:
    name: str
    params: str
    relative_speed: int  # Higher is faster


DEFAULT_WHISPER_MODEL = ModelSpec(
    name="medium.en", params="769M", relative_speed=2)


@dataclasses.dataclass
class JobSpec:
    video_id: str
    whisper_model: ModelSpec = DEFAULT_WHISPER_MODEL
    bucket_for_upload_name: str = 'holosays'

    def yt_video_url(self) -> str:
        return 'https://www.youtube.com/watch?v=' + self.video_id


YT_DLP_DOWNLOAD_FILE_TEMPL = '%(id)s.%(ext)s'


CACHE_DIR = '/cache'

# Location of modal checkpoint.
MODEL_DIR = pathlib.Path(CACHE_DIR, "model")


volume = modal.SharedVolume().persist('vtscribe-cache-vol')

if stub.is_inside():
    from loguru import logger  # TODO: Replace with Python Logger


@stub.function(timeout=40000, secret=modal.Secret.from_name("vtscribe-secrets"))
@stub.web_endpoint(method="POST", wait_for_response=True)
def transcribe(video_id: str):
    return transcribe_vod.call(JobSpec(video_id=video_id))


@stub.function(image=app_image, shared_volumes={CACHE_DIR: volume}, timeout=3000)
def transcribe_vod(job_spec: JobSpec):
    """Main worker function for VOD transcription.

    Args:
        job_spec (JobSpec): A job spec describing what needs to be done.

    Returns:
        _type_: _description_
    """
    download_audio.call(job_spec.yt_video_url())
    input_raw_file_path: str = f"{CACHE_DIR}/{job_spec.video_id}.mp3"
    result_metadata = do_transcribe.call(
        input_raw_file_path, model=job_spec.whisper_model, result_path=f"{CACHE_DIR}/{job_spec.video_id}-result.json")
    upload_to_obj_storage.call(
        result_metadata['result_file_name'], result_metadata['result_path'])

    return JSONResponse(content=result_metadata, status_code=200)


@stub.function(image=app_image, shared_volumes={CACHE_DIR: volume}, timeout=3000)
def download_audio(yt_video_url: str):
    """Downloads audio track for a given Youtube video URL as an mp3.

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
        error_code = ydl.download(yt_video_url)
        logger.info("Finished downloading with code: {}", error_code)


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
def do_transcribe(input_audio_file_path: str, model: ModelSpec, result_path: str = "result.json") -> dict:
    """Perform transcription of an audio file using Whisper.
    Returns a dict of metadata of the result

    Args:
        input_audio_file_path (str): path to mp3 audio file to transcribe
        model (ModelSpec): Whisper model to use for transcription
        result_path (str): path to output results file, must be JSON
    """
    import ffmpeg
    import time
    import os

    start_time = time.time()

    output_file = input_audio_file_path.replace(".mp3", ".wav")
    logger.info("Downsampling file {} --> {}",
                input_audio_file_path, output_file)
    stream = ffmpeg.input(input_audio_file_path)
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
        # TODO: Trim numeric token values from segment
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
                input_audio_file_path, exec_time)

    metadata = {'result_path': result_path,
                'result_file_name': os.path.basename(result_path),
                'src_audio_file_name': input_audio_file_path}

    logger.info("Metadata: {} cwd: {}", metadata, os.getcwd())

    return metadata

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


@stub.function(image=app_image, shared_volumes={CACHE_DIR: volume}, timeout=3000)
def upload_to_obj_storage(bucket_path: str, local_file_path: str, bucket_name=DO_DEFAULT_BUCKET_NAME):
    import boto3
    import os

    logger.info("Uploading to bucket: {} to: {} from: {}",
                bucket_name, bucket_path, local_file_path)

    s3 = boto3.resource('s3',
                        endpoint_url='https://ams3.digitaloceanspaces.com',
                        aws_access_key_id=os.environ["DO_SPACE_HOLOSAYS_KEY"],
                        aws_secret_access_key=os.environ["DO_SPACE_HOLOSAYS_SECRET"])

    # Upload the JSON file to the Space
    s3.Object(bucket_name, bucket_path).put(Body=open(local_file_path, 'rb'))
