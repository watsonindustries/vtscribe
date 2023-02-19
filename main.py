"""
Microservice worker that uses OpenAI's Whisper model to download and transcribe videos.
"""
import modal

app_image = (
    modal.Image.debian_slim()
    .pip_install(
        "ffmpeg-python",
        "loguru==0.6.0",
        "yt-dlp"
    )
    .apt_install("ffmpeg")
    .pip_install("ffmpeg-python")
)

stub = modal.Stub("vtscribe", image=app_image)

YT_DLP_DOWNLOAD_DIR = '/root/vtscribe/downloads'
YT_DLP_DOWNLOAD_FILE_TEMPL = '%(id)s.%(ext)s'

WHISPER_SRC_FOLDER = '/root/vtscribe/audio'

CACHE_DIR = '/root/vtscribe'

volume = modal.SharedVolume().persist('vtscribe-cache-vol')

if stub.is_inside():
    from loguru import logger    
    import os

@stub.local_entrypoint
def main():
    import os
    video_id = 'IxiUmUXPGkw'
    download_audio.call('https://www.youtube.com/watch?v=' + video_id)
    print('Directories in main: ', os.listdir())
    input_file_name = f"vtscribe/{video_id}.mp3"
    raw_file_name = f"{video_id}.wav"
    
    downsample.call(input_file_name, raw_file_name)

@stub.function(image=app_image, shared_volumes={CACHE_DIR: volume})
def download_audio(url: str):
    import yt_dlp
    import os
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': f"vtscribe/{YT_DLP_DOWNLOAD_FILE_TEMPL}"
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        error_code = ydl.download(url)
    print('Directories in download_audio: ', os.listdir())

@stub.function(image=app_image, shared_volumes={CACHE_DIR: volume})
def downsample(input_file: str, output_file: str):
    """Downsamples the input file to the correct sampling rate for Whisper.
    """
    import ffmpeg
    import os
    print('Directories in downsample: ', os.listdir())
    logger.info("Downsampling file {} --> {}", input_file, output_file)
    stream = ffmpeg.input(input_file)
    stream = ffmpeg.output(stream, output_file, **{'ar': '16000','acodec':'pcm_s16le'})
    ffmpeg.run(stream)

    logger.info("Finished downsampling file: {}", output_file)
