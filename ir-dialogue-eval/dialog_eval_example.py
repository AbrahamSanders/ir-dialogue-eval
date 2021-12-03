import argparse
import pandas as pd
import matplotlib.pyplot as plt

from dialog_eval import DialogEval
from domain import Domain

parser = argparse.ArgumentParser("Test dialogue evaluation against Elasticsearch index.")
parser.add_argument("--elasticsearch-uri", default="http://localhost:9200", required=False,
                    help="URI of elasticsearch server. (default: %(default)s)")
parser.add_argument("--elasticsearch-index", default="dialog-eval", required=False,
                    help="Name of the elasticsearch index. (default: %(default)s)")
parser.add_argument("--embedding-model", default="all-MiniLM-L12-v2", required=False,
                    help="Name of the sentence-transformers embedding model. (default: %(default)s)")

args = parser.parse_args()

print()
print("Running with arguments:")
print(args)
print()

dialogeval = DialogEval(args.elasticsearch_uri, args.elasticsearch_index,
                        embedding_model=args.embedding_model)

# Example 1
context = ["Oh my god, she's got a gun!",
           "Don't move, I'll blow your head off!"]
utterances = [
    "Please don't kill me!",
    "Ahhh! Somebody help!",
    "Go ahead, shoot me.",
    "Right between the eyes please."
]

utterances_in_context = [context + [u] for u in utterances]
reference_utterances = ["I don't want to die!"] * len(utterances)

scores_zero_ref = dialogeval.score_utterances(utterances_in_context)
scores_single_ref = dialogeval.score_utterances(utterances_in_context, reference_utterances)

scores_df = (
    pd.DataFrame({"response": utterances, "zero_ref": scores_zero_ref, "single_ref": scores_single_ref})
)
scores_df.plot.barh(x="response").invert_yaxis()
plt.xlabel("score")

# Example 2
context = ["Hi! How are you?",
           "I need a room to stay for the night."]

utterances = [
    "What happened to you?",
    "Sorry, I can't help you.",
    "Sure, the Ridge Hotel is rated 9/10 and has free Wifi.",
]

utterances_in_context = [context + [u] for u in utterances]

scores_general = dialogeval.score_utterances(utterances_in_context, domains=[Domain.GENERAL.value])
scores_hotels = dialogeval.score_utterances(utterances_in_context, domains=[Domain.HOTELS.value])

scores_df = (
    pd.DataFrame({"response": utterances, "general": scores_general, "hotels": scores_hotels})
)
scores_df.plot.barh(x="response").invert_yaxis()
plt.xlabel("score")

# Example 3

utterances_in_context = [
    ["Hi! How are you?",
    "Where the hell have you been?",
    "Working hard."]
]

scores, details = dialogeval.score_utterances(utterances_in_context, return_details=True)
details_df = (
    pd.DataFrame(details[0])[["nearest_utterances", "weighted_cosines"]]
      .sort_values(by="weighted_cosines", ascending=False)
)

