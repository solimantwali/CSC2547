import logging
import os

logger = logging.getLogger(__name__)


MODEL_COUNSELLOR = "gpt-4-0125-preview"
MODEL_CLIENT = "gpt-4-0125-preview"
MODEL_EVALUATOR = "gpt-4-0125-preview"
MODEL_END_CLS = "gpt-3.5-turbo-0125"
PRICE_LIMIT = 5
TURN_LIMIT = 100
CLIENT_VER = 0
COUNSELLOR_VER = 0
EVALUATOR_VER = 1
EVALUATOR_CSV_VER = 4
EVALUATOR_GLOBAL_SCORE_VER = 3
END_CLASSIFIER_VER = 0
PERSIST_DIR = "data/csv_transcript_persist_miti"


try:
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    OPENAI_ORG_ID = os.environ["OPENAI_ORG_ID"]
except KeyError:
    logger.error("Either OPENAI_API_KEY or OPENAI_ORG_ID not found in the environment")
