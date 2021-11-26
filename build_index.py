import logging
import argparse
import sys
from glob import glob

from dataset_loaders.dataset_loader_factory import get_dataset_loader
from dataset_importer import DatasetImporter

parser = argparse.ArgumentParser("Run analytics on Synced Entities in Elasticsearch")
parser.add_argument("--logfile", default="build_index_log.txt", required=False, 
                    help="Path to the log file to write to. (default: %(default)s)")
parser.add_argument("--loglevel", default="WARNING", required=False, 
                    help="Logging level. (default: %(default)s)")
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

#Configure logging
file_handler = logging.FileHandler(filename=args.logfile)
stdout_handler = logging.StreamHandler(sys.stdout)
logging.basicConfig(handlers = [file_handler, stdout_handler], 
                    format="[%(asctime)s - %(levelname)s]: %(message)s", 
                    level=logging.getLevelName(args.loglevel))
print("Logging level set to {0}...".format(args.loglevel))
print()

#Initialize the importer, which internally initializes the elasticsearch index
#and embedding models
importer = DatasetImporter(args.elasticsearch_uri, args.elasticsearch_index,
                           embedding_model=args.embedding_model)

#Import all datasets
datasets = glob("datasets/*")
for dataset in datasets:
    loader = get_dataset_loader(dataset)
    ids, dialogs, domains = loader.load_dataset()
    importer.import_dataset(loader.dataset_name, ids, dialogs, domains)