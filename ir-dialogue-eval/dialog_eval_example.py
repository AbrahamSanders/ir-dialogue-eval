import argparse
from dialog_eval import DialogEval

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

utterances_in_context = [
    ["Hello! How are you?",
     "Not good.",
     "What's wrong?"]
]

reference_utterances = [
    ["What's wrong with you anyway?", "Sorry to hear it."]
]

scores = dialogeval.score_utterances(utterances_in_context, reference_utterances)

print()
print()
print(scores)