import os
import logging
from logging.config import dictConfig

def configure_logger(name, log_path):
    try:
        log_dir, log_file = os.path.split(log_path)
    except Exception as e:
        raise ValueError("log_path error, MUST be absolute path")
    dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'default': {
                'format': '%(asctime)s - %(module)s[:%(lineno)d] - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
                }
        },
        'handlers': {
            'console_handler': {
                'level': logging.DEBUG,
                'class': 'logging.StreamHandler',
                'formatter': 'default',
                'stream': 'ext://sys.stdout'
            },
            'info_handler': {
                'level': logging.INFO,
                'class': 'logging.handlers.TimedRotatingFileHandler',
                'formatter': 'default',
                'when': 'D',
                'interval': 1,
                'filename': os.path.join(log_dir, log_file),
                'backupCount': 7,
                "encoding": "utf8"
            },
            'debug_handler': {
                'level': logging.DEBUG,
                'class': 'logging.handlers.TimedRotatingFileHandler',
                'formatter': 'default',
                'when': 'D',
                'interval': 1,
                'filename': os.path.join(log_dir, log_file),
                'backupCount': 7,
                "encoding": "utf8"
            },
            "error_handler": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": logging.ERROR,
                "formatter": "default",
                "filename": os.path.join(log_dir, log_file),
                "maxBytes": 10485760,
                "backupCount": 7,
                "encoding": "utf8"
                }
        },
        'loggers': {
            '': {  # root logger
                'level': logging.DEBUG,
                'handlers': ['console_handler', 'error_handler'],
            },
            'debug': {
                'level': logging.DEBUG,
                'handlers': ['debug_handler', 'error_handler'],
                'propagate': True
            },
            'info': {
                'level': logging.INFO,
                'handlers': ['info_handler', 'error_handler'],
                'propagate': True
            }
        }
    })
    return logging.getLogger(name)
