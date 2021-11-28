import argparse
from dialog_eval import DialogEval

parser = argparse.ArgumentParser("Test dialogue evaluation against Elasticsearch index.")
parser.add_argument("--elasticsearch-uri", default="http://localhost:9200", required=False,
                    help="URI of elasticsearch server. (default: %(default)s)")
parser.add_argument("--elasticsearch-index", default="dialog-eval", required=False,
                    help="Name of the elasticsearch index. (default: %(default)s)")
parser.add_argument("--embedding-model", default="all-mpnet-base-v2", required=False,
                    help="Name of the sentence-transformers embedding model. (default: %(default)s)")

args = parser.parse_args()

print()
print("Running with arguments:")
print(args)
print()


dialogeval = DialogEval(args.elasticsearch_uri, args.elasticsearch_index,
                        embedding_model=args.embedding_model)

contexts = [
    ["Hello there! I have 12 days off starting on August 22, and I'm leaving from Santo Domingo",
     "Do you have a budget?",
     "Not a strict budget",
     "Are you travelling with anyone?"
     ]
]

utterances = [
    "3 people."
]

scores = dialogeval.score_utterances(contexts, utterances)

print()
print()
print(scores)