import dataclasses
import logging
import pathlib
from typing import Optional, Set


def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(levelname)s: %(asctime)s: %(name)s  %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False  # Prevent the modal client from double-logging.
    return logger


@dataclasses.dataclass
class ModelSpec:
    name: str
    params: str
    relative_speed: int  # Higher is faster


SUPPORTED_WHISPER_MODELS = {
    "tiny.en": ModelSpec(name="tiny.en", params="39M", relative_speed=32),
    # Takes around 3-10 minutes to transcribe a podcast, depending on length.
    "base.en": ModelSpec(name="base.en", params="74M", relative_speed=16),
    "small.en": ModelSpec(name="small.en", params="244M", relative_speed=6),
    "medium.en": ModelSpec(name="medium.en", params="769M", relative_speed=2),
    # Very slow. Will take around 45 mins to 1.5 hours to transcribe.
    "large": ModelSpec(name="large", params="1550M", relative_speed=1),
}

DEFAULT_WHISPER_MODEL = SUPPORTED_WHISPER_MODELS["medium.en"]


@dataclasses.dataclass
class Source:
    id: Optional[str]
    url: Optional[str]
    type: str = 'yt'

    @staticmethod
    def validate_type(valid_types: Set[str], source_type: str):
        if source_type not in valid_types:
            raise ValueError(
                f"Invalid type '{source_type}', must be one of {valid_types}")

    def validate(self):
        if self.url is None:
            if not (self.id and self.type):
                raise ValueError("Both id and type must be present")

        self.validate_type({'yt'}, self.type)

        return self


@dataclasses.dataclass
class JobSpec:
    source: Source
    whisper_model: ModelSpec = DEFAULT_WHISPER_MODEL
    bucket_for_upload: str = 'holosays'

    def yt_video_url(self) -> str:
        source = self.source
        return ('https://www.youtube.com/watch?v=' + source.id) if source.id else source.url


YT_DLP_DOWNLOAD_FILE_TEMPL = '%(id)s.%(ext)s'

CACHE_DIR = '/cache'

# Location of modal checkpoint.
MODEL_DIR = pathlib.Path(CACHE_DIR, "model")

MODAL_STUB_NAME = 'vtscribe'
MODAL_SECRETS_NAME = 'vtscribe-secrets'

DO_DEFAULT_BUCKET_NAME = 'holosays'
S3_ENDPOINT_URL = 'https://ams3.digitaloceanspaces.com'
