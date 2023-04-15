"""
Microservice worker that uses OpenAI's Whisper model to download and transcribe videos.
"""
import modal
import pathlib
import json
from typing import Iterator, Tuple
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

import config
import util

app_image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg", "git")
    .pip_install(
        "boto3",
        "ffmpeg-python",
        "torchaudio==0.12.1",
        # Youtube API keeps changing frequently, so we want the latest and greatest
        "git+https://github.com/yt-dlp/yt-dlp.git@master",
        "https://github.com/openai/whisper/archive/9f70a352f9f8630ab3aa0d06af5cb9532bd8c21d.tar.gz"
    )
)

auth_scheme = HTTPBearer()

stub = modal.Stub(config.MODAL_STUB_NAME, image=app_image, secrets=[
                  modal.Secret.from_name(config.MODAL_SECRETS_NAME)])

logger = config.get_logger(__name__)
volume = modal.SharedVolume().persist('vtscribe-cache-vol')

# The main entrypoint for the app


@stub.function(timeout=40000, secret=modal.Secret.from_name("vtscribe-secrets"))
@stub.web_endpoint(method="POST", wait_for_response=False)
async def transcribe(request: Request, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    import os

    if token.credentials != os.environ["VTSCRIBE_API_TOKEN"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    source_id = request.query_params.get('source_id', None)
    source_url = request.query_params.get('source_url', None)
    source_type = request.query_params.get('source_type', 'yt')
    model_name = request.query_params.get('model_name', 'small.en')

    source = config.Source(source_id, source_url, source_type).validate()
    return transcribe_vod.call(config.JobSpec(source=source, whisper_model=config.SUPPORTED_WHISPER_MODELS[model_name]))


@stub.function(image=app_image, shared_volumes={config.CACHE_DIR: volume}, timeout=3000)
def transcribe_vod(job_spec: config.JobSpec):
    """Main worker function for VOD transcription.

    Args:
        job_spec (JobSpec): A job spec describing what needs to be done.

    Returns:
        _type_: _description_
    """

    import whisper

    source = job_spec.source

    # pre-download the model to the cache path, because the _download fn is not
    # thread-safe.
    whisper._download(
        whisper._MODELS[job_spec.whisper_model.name], config.MODEL_DIR, False)

    download_audio.call(job_spec.yt_video_url() if source.type ==
                        'yt' else job_spec.twitch_video_url())
    input_raw_file_path: str = f"{config.CACHE_DIR}/{source.id}.mp3"
    result_metadata = do_transcribe.call(
        input_raw_file_path, model=job_spec.whisper_model, result_path=f"{config.CACHE_DIR}/{source.id}-{source.type}-result.json")

    upload_to_obj_storage.call(
        f"transcripts/{result_metadata['result_file_name']}", result_metadata['result_path']
    )
    upload_to_obj_storage.call(
        f"transcripts/{result_metadata['result_vtt_name']}", result_metadata['result_vtt_path']
    )

    gc_artefacts.call()


@stub.function(image=app_image, shared_volumes={config.CACHE_DIR: volume}, timeout=3000)
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
        'outtmpl': config.CACHE_DIR + '/' + config.YT_DLP_DOWNLOAD_FILE_TEMPL
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        error_code = ydl.download(yt_video_url)
        logger.info(f"Finished downloading with code: {error_code}")


@stub.function(
    image=app_image,
    shared_volumes={config.CACHE_DIR: volume},
    cpu=2,
    timeout=3000
)
def transcribe_segment(
    start: float,
    end: float,
    audio_filepath: pathlib.Path,
    model: config.ModelSpec,
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
            model.name, device=device, download_root=config.MODEL_DIR
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


@stub.function(image=app_image, shared_volumes={config.CACHE_DIR: volume}, timeout=30000)
def do_transcribe(input_audio_file_path: str, model: config.ModelSpec, result_path: str = "result.json") -> dict:
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

    # TODO: Put downsampling into separate function
    output_file = input_audio_file_path.replace(".mp3", ".wav")
    logger.info(f"Downsampling file {input_audio_file_path} --> {output_file}")
    stream = ffmpeg.input(input_audio_file_path)
    stream = ffmpeg.output(stream, output_file, **
                           {'ar': '16000', 'acodec': 'pcm_s16le'})
    ffmpeg.run(stream, overwrite_output=True, quiet=True)

    logger.info(f"Finished downsampling file: {output_file}")
    logger.info(f"Preparing to transcribe file {output_file}")

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

    end_time = time.time()

    logger.info(f"Writing openai/whisper transcription to {result_path}")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=4)

    result_vtt_path = result_path.replace(".json", ".vtt")
    logger.info(f"Writing VTT file to {result_vtt_path}")
    with open(result_vtt_path, 'w', encoding='utf-8') as vtt:
        util.write_vtt(result['segments'], file=vtt)

    exec_time = end_time - start_time

    logger.info(
        f"Finished transcribing file {input_audio_file_path} in {exec_time} seconds")

    metadata = {
        'result_path': result_path,
        'result_vtt_path': result_vtt_path,
        'result_file_name': os.path.basename(result_path),
        'result_vtt_name': os.path.basename(result_vtt_path),
        'src_audio_file_name': input_audio_file_path
    }

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


@stub.function(image=app_image, shared_volumes={config.CACHE_DIR: volume}, timeout=3000)
def upload_to_obj_storage(bucket_path: str, local_file_path: str, bucket_name=config.DO_DEFAULT_BUCKET_NAME):
    import boto3
    import os

    logger.info(
        f"Uploading to bucket: {bucket_name} to: {bucket_path} from: {local_file_path}")

    s3 = boto3.resource('s3',
                        endpoint_url=config.S3_ENDPOINT_URL,
                        aws_access_key_id=os.environ["DO_SPACE_HOLOSAYS_KEY"],
                        aws_secret_access_key=os.environ["DO_SPACE_HOLOSAYS_SECRET"])

    # Upload the JSON file to the Space
    s3.Object(bucket_name, bucket_path).put(
        Body=open(local_file_path, 'rb'), ACL='public-read')


@stub.function(image=app_image, shared_volumes={config.CACHE_DIR: volume})
def gc_artefacts():
    """Garbage collect any remains in the cache directory.
    """
    import os

    directory = config.CACHE_DIR

    logger.info(f"Garbage collecting artefacts in {directory}")

    for filename in os.listdir():
        if filename.endswith('.mp3') or filename.endswith('.wav') or filename.endswith('.json'):
            os.remove(os.path.join(directory, filename))
