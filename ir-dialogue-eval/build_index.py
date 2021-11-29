import logging
import argparse
import sys
from glob import glob
from os import path

from dataset_loaders.dataset_loader_factory import get_dataset_loader
from dialog_eval import DialogEval

parser = argparse.ArgumentParser("Build Elasticsearch index for dialogue evaluation.")
parser.add_argument("--logfile", default="build_index_log.txt", required=False, 
                    help="Path to the log file to write to. (default: %(default)s)")
parser.add_argument("--loglevel", default="WARNING", required=False, 
                    help="Logging level. (default: %(default)s)")
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

#Configure logging
file_handler = logging.FileHandler(filename=args.logfile)
stdout_handler = logging.StreamHandler(sys.stdout)
logging.basicConfig(handlers = [file_handler, stdout_handler], 
                    format="[%(asctime)s - %(levelname)s]: %(message)s", 
                    level=logging.getLevelName(args.loglevel))
print("Logging level set to {0}...".format(args.loglevel))
print()

#Initialize the dialog evaluator, which internally initializes the elasticsearch index
#and embedding models
dialogeval = DialogEval(args.elasticsearch_uri, args.elasticsearch_index,
                        embedding_model=args.embedding_model)

#Import all datasets
datasets_path = "datasets"
if not path.isdir(datasets_path):
    datasets_path = path.join("..", datasets_path)
glob_path = path.join(datasets_path, "*")

datasets = glob(glob_path)
for dataset in datasets:
    loader = get_dataset_loader(dataset)
    ids, dialogs, domains = loader.load_dataset()
    dialogeval.import_dataset(loader.dataset_name, ids, dialogs, domains)