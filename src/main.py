"""Entrypoint to emulate conversation between chatbots"""
import argparse
import logging
from pathlib import Path

from openai import OpenAI

from .bots.ClientBot import ClientBot
from .bots.CounsellorBot import CounsellorBot
from .bots.EndClassifierBot import EndClassifierBot
from .bots.EvalBot import EvalBot
from .constants import CLIENT_VER
from .constants import COUNSELLOR_VER
from .constants import END_CLASSIFIER_VER
from .constants import EVALUATOR_CSV_VER
from .constants import EVALUATOR_GLOBAL_SCORE_VER
from .constants import EVALUATOR_VER
from .constants import OPENAI_API_KEY
from .constants import OPENAI_ORG_ID
from .constants import PRICE_LIMIT
from .constants import TURN_LIMIT
from .conversation import SimpleConversation


logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--client_name", required=False, type=str, default="Axiom")
parser.add_argument(
    "--evaluate_realtime", required=False, action="store_true", default=False
)
parser.add_argument("--mode", required=False, type=str, default="normal")
parser.add_argument("--folder", required=False, type=str)
parser.add_argument("--few_shot", action="store_true", default=False)
parser.add_argument("--counsellor_sample", action="store_true", default=False)

args = parser.parse_args()

if __name__ == "__main__":
    data_folder = Path("data/")
    assert data_folder.exists() and data_folder.is_dir()
    assert args.mode in ["normal", "turbo", "eval", "eval_csv", "global_scores"]
    if args.mode == "eval":
        folder = args.folder
        transcript_json = open(Path(folder) / "counsellor.json")
        output_dir = Path(folder)
    elif args.mode == "eval_csv":
        folder = args.folder
        transcript_csv = open(
            Path(folder) / "counsellor_bc_history.csv", encoding="MacRoman"
        )
        output_dir = Path(folder)
        print("eval csv")
    elif args.mode == "global_scores":
        folder = args.folder
        transcript_json = open(Path(folder) / "counsellor.json")
        output_dir = Path(folder)
    else:
        new_dir_name = f"simple-conversation-{len(list(data_folder.iterdir()))+1:03d}"
        output_dir = data_folder / new_dir_name
        output_dir.mkdir(exist_ok=True)

    client_openai_obj = OpenAI(organization=OPENAI_ORG_ID, api_key=OPENAI_API_KEY)
    counsellor_openai_obj = OpenAI(organization=OPENAI_ORG_ID, api_key=OPENAI_API_KEY)
    evaluator_openai_obj = (
        OpenAI(organization=OPENAI_ORG_ID, api_key=OPENAI_API_KEY)
        if args.evaluate_realtime or args.mode in ["eval", "eval_csv", "global_scores"]
        else None
    )
    end_classifier_openai_obj = (
        OpenAI(organization=OPENAI_ORG_ID, api_key=OPENAI_API_KEY)
        if args.mode == "turbo"
        else None
    )

    client = ClientBot(CLIENT_VER)
    counsellor = CounsellorBot(COUNSELLOR_VER, args.client_name, args.counsellor_sample)

    evaluator = None
    match args.mode:
        case "eval_csv":
            evaluator = EvalBot(EVALUATOR_CSV_VER, few_shot=args.few_shot)
        case "eval":
            evaluator = EvalBot(EVALUATOR_VER, few_shot=args.few_shot)
        case "global_scores":
            evaluator = EvalBot(EVALUATOR_GLOBAL_SCORE_VER, few_shot=args.few_shot)
        case _:
            evaluator = (
                EvalBot(EVALUATOR_CSV_VER, few_shot=args.few_shot)
                if args.evaluate_realtime
                else None
            )

    end_classifier = (
        EndClassifierBot(END_CLASSIFIER_VER) if args.mode == "turbo" else None
    )

    conversation = SimpleConversation(
        client_bot=client,
        counsellor_bot=counsellor,
        eval_bot=evaluator,
        end_classifier=end_classifier,
        output_dir=output_dir,
        mode=args.mode,
        price_limit=PRICE_LIMIT,
        turn_limit=TURN_LIMIT,
    )

    if args.mode == "eval":
        conversation.evaluate(
            evaluator_openai_obj=evaluator_openai_obj,
            transcript_json=transcript_json,
        )
    elif args.mode == "eval_csv":
        conversation.evaluate_csv(
            evaluator_openai_obj=evaluator_openai_obj, transcript_csv=transcript_csv
        )
    elif args.mode == "global_scores":
        conversation.evaluate_globalscores(
            evaluator_openai_obj=evaluator_openai_obj, transcript_json=transcript_json
        )
    else:
        conversation.converse(
            client_openai_obj=client_openai_obj,
            counsellor_openai_obj=counsellor_openai_obj,
            evaluator_openai_obj=evaluator_openai_obj,
            end_classifier_openai_obj=end_classifier_openai_obj,
        )
