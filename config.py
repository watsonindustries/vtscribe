import dataclasses
import pathlib


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
    source_type: str = 'yt'
    whisper_model: ModelSpec = DEFAULT_WHISPER_MODEL
    bucket_for_upload_name: str = 'holosays'

    def yt_video_url(self) -> str:
        return 'https://www.youtube.com/watch?v=' + self.video_id
      
YT_DLP_DOWNLOAD_FILE_TEMPL = '%(id)s.%(ext)s'

CACHE_DIR = '/cache'

# Location of modal checkpoint.
MODEL_DIR = pathlib.Path(CACHE_DIR, "model")

MODAL_STUB_NAME = 'vtscribe'
MODAL_SECRETS_NAME = 'vtscribe-secrets'

DO_DEFAULT_BUCKET_NAME = 'holosays'
