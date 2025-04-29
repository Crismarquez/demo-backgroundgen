from pathlib import Path
import sys
import os
import logging
import logging.config

from rich.logging import RichHandler
from dotenv import dotenv_values, load_dotenv


BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
LOGS_DIR = Path(BASE_DIR, "logs")
DATA_DIR = Path(BASE_DIR, "data")
EXPERIMENTS_DIR = Path(DATA_DIR, "experiments")
RUNS_BATCH_DIR = Path(DATA_DIR, "runs_batch")   
RUNS_INFERENCE_DIR = Path(DATA_DIR, "runs_inference")

LOGS_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
RUNS_BATCH_DIR.mkdir(parents=True, exist_ok=True)
RUNS_INFERENCE_DIR.mkdir(parents=True, exist_ok=True)

ENV_VARIABLES = {
    **dotenv_values(BASE_DIR/".env"),  # load environment variables from .env file
    **os.environ,  # load environment variables from the system
}
load_dotenv(dotenv_path=BASE_DIR / ".env")

N_IMAGES = 2

# Logger
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "minimal": {"format": "%(message)s"},
        "detailed": {
            "format": "%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "minimal",
            "level": logging.DEBUG,
        },
        "info": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "info.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.INFO,
        },
        "error": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "error.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.ERROR,
        },
    },
    "root": {
        "handlers": ["console", "info", "error"],
        "level": logging.INFO,
        "propagate": True,
    },
}

logging.config.dictConfig(logging_config)
logger = logging.getLogger()
logger.handlers[0] = RichHandler(markup=True)