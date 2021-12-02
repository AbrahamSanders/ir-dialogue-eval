import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress, spearmanr

from dialog_eval import DialogEval
from utils import clean_up_tokenization

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

human_ratings_df = pd.read_csv("../test_data/mturk_rating_processed_output.csv")

contexts = [list(map(clean_up_tokenization, c.split("||||"))) for c in human_ratings_df["context"]]
responses = list(map(clean_up_tokenization, human_ratings_df["response"]))
utterances_in_context = [c + [u] for c,u in zip(contexts, responses)]
reference_utterances_single = list(map(clean_up_tokenization, human_ratings_df["prevgt"]))
reference_utterances_multi = [list(map(clean_up_tokenization, r.split("\t"))) for r in human_ratings_df["all_references"]]
human_scores = human_ratings_df["human_average_rating"].to_numpy()
models = human_ratings_df["model"].to_numpy()

for n_samples in [0, 4, 25, 50, 100]:
    
    for reference_utterances in [None, reference_utterances_single, reference_utterances_multi]:
        if n_samples == 0 and reference_utterances is None:
            continue
        
        n_refs = (0 if reference_utterances is None 
                  else 1 if reference_utterances is reference_utterances_single
                  else len(reference_utterances[0]))
        
        scores = dialogeval.score_utterances(utterances_in_context, reference_utterances, n_samples=n_samples)
         
        #Utterance level
        linreg = linregress(human_scores, scores)
        spearman = spearmanr(human_scores, scores)
        
        plt.scatter(human_scores, scores)
        plt.plot(human_scores, linreg.slope*human_scores + linreg.intercept, 
                 color="red", linestyle="dashed")
        plt.xlabel("Average Rating")
        plt.ylabel("IR-Dialogue-Eval Score")
        plt.title("Human ratings vs. IR-Dialogue-Eval scores \n (Utterance-level; samples={}; refs={}) \n "
                  "Pearson $r={:.3f};\\ p={:.3f}$ \n Spearman $\\rho={:.3f};\\ p={:.3f}$.".format(
                    n_samples, n_refs, linreg.rvalue, linreg.pvalue, spearman.correlation, spearman.pvalue))
        plt.show()
        
        #Model level
        unique_models = np.unique(models)
        model_human_scores = np.array([np.mean(human_scores[models==m]) for m in unique_models])
        model_scores = np.array([np.mean(scores[models==m]) for m in unique_models])
        
        linreg = linregress(model_human_scores, model_scores)
        spearman = spearmanr(model_human_scores, model_scores)
        
        plt.scatter(model_human_scores, model_scores)
        plt.plot(model_human_scores, linreg.slope*model_human_scores + linreg.intercept, 
                 color="red", linestyle="dashed")
        for i in range(len(unique_models)):
            plt.text(model_human_scores[i]+0.05, model_scores[i], unique_models[i])
        plt.xlabel("Average Rating")
        plt.ylabel("IR-Dialogue-Eval Score")
        plt.title("Human ratings vs. IR-Dialogue-Eval scores \n (System-level; samples={}; refs={}) \n "
                  "Pearson $r={:.3f};\\ p={:.3f}$ \n Spearman $\\rho={:.3f};\\ p={:.3f}$.".format(
                    n_samples, n_refs, linreg.rvalue, linreg.pvalue, spearman.correlation, spearman.pvalue))
        plt.show()
