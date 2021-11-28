import numpy as np
import math
import itertools
from tqdm import trange
from elasticsearch.helpers import bulk
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist

from index_setup import verify_and_configure_index

class DialogEval(object):
    def __init__(self, elasticsearch_uri, elasticsearch_index_name="dialog-eval", 
                 elasticsearch_timeout_secs=60, elasticsearch_chunk_size=100, 
                 embedding_model="all-mpnet-base-v2", normalize_embeddings=True,
                 embed_batch_size=32, process_batch_size=128, show_progress=True):
        
        #Initialize elasticsearch settings
        self.es = Elasticsearch(hosts=[elasticsearch_uri], timeout=elasticsearch_timeout_secs)
        self.es_index_name = elasticsearch_index_name
        self.es_chunk_size = elasticsearch_chunk_size
        
        #Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        embedding_dims = self.embedding_model[0].get_word_embedding_dimension()
        
        #Verify or setup elasticsearch index
        verify_and_configure_index(self.es, self.es_index_name, embedding_dims)
        
        #Initialize other processing settings
        self.normalize_embeddings = normalize_embeddings
        self.embed_batch_size = embed_batch_size
        self.process_batch_size = process_batch_size
        self.show_progress = show_progress
    
    def score_utterances(self, contexts, utterances, reference_utterances=None, domains=None, 
                         source_datasets=None, source_speakers=None, max_samples=30):
        scores = []
        n_samples = max_samples if reference_utterances is None else max_samples-1
        for i in trange(len(utterances), desc="Utterances", disable=not self.show_progress):
            #Embed everything
            pooled_context_embedding = None
            if contexts[i]:
                context_embeddings = self.embedding_model.encode(contexts[i], batch_size=self.embed_batch_size, 
                                                                 normalize_embeddings=self.normalize_embeddings, 
                                                                 show_progress_bar=False)
                pooled_context_embedding = np.mean(context_embeddings, axis=0)
            
            utterance_embedding = self.embedding_model.encode(utterances[i], 
                                                              normalize_embeddings=self.normalize_embeddings, 
                                                              show_progress_bar=False)
            
            if reference_utterances is not None:
                ref_utterance_embedding = self.embedding_model.encode(reference_utterances[i], 
                                                                      normalize_embeddings=self.normalize_embeddings, 
                                                                      show_progress_bar=False)
                
            #Get nearest neighbor contexts from Elasticsearch
            scoring_query = self._get_scoring_query(pooled_context_embedding, domains, source_datasets, source_speakers)
            results = self.es.search(scoring_query, self.es_index_name, size=n_samples)
            results = [{"_score": hit["_score"], **hit["_source"]} for hit in results["hits"]["hits"]]
            
            if contexts[i]:
                context_cosines = np.array([r["_score"] for r in results])-1.0 # undo +1.0 in ES script score
            else:
                context_cosines = np.ones(len(results), dtype=np.float64)
            nearest_utterances = [r["utterance"] for r in results]
            nearest_utterance_embeddings = np.vstack([r["utterance_embedding"] for r in results]).astype(np.float32)
            
            if reference_utterances is not None:
                context_cosines = np.concatenate((context_cosines, [1.0]))
                nearest_utterances.append(reference_utterances[i])
                nearest_utterance_embeddings = np.vstack((nearest_utterance_embeddings, ref_utterance_embedding))
                
            #Compute average utterance similarity weighted by context similarity
            utterance_cosines = 1 - np.squeeze(cdist(nearest_utterance_embeddings, 
                                                     np.expand_dims(utterance_embedding, 0), 
                                                     metric="cosine"), axis=1)
            score = np.average(utterance_cosines, weights=context_cosines)
            scores.append(score)
        
        return np.array(scores)
                
    def import_dataset(self, dataset_name, ids, dialogs, domains):
        n_batches = math.ceil(len(ids) / self.process_batch_size)
        for i in trange(n_batches, desc="Dialog Batches (%s)" % dataset_name, disable=not self.show_progress):
            start = i * self.process_batch_size
            end = start + self.process_batch_size
            batch_ids = ids[start:end]
            batch_dialogs = dialogs[start:end]
            batch_domains = domains[start:end]
            
            batch_utterances = [utterance[1] for utterance in itertools.chain(*batch_dialogs)]
            batch_embeddings = self.embedding_model.encode(batch_utterances, batch_size=self.embed_batch_size, 
                                                           normalize_embeddings=self.normalize_embeddings, 
                                                           show_progress_bar=False)
            es_docs = []
            embeddings_idx = 0
            for j, dialog in enumerate(batch_dialogs):
                dialog_length = len(dialog)
                
                dialog_embeddings = batch_embeddings[embeddings_idx:(embeddings_idx + dialog_length)]
                embeddings_idx += dialog_length
                
                #construct utterance documents
                for k in range(dialog_length):
                    utterance_id = "%s_%s_%s" % (dataset_name, batch_ids[j], k+1)
                    source = {
                        "utterance_id": utterance_id,
                        "utterance_embedding": dialog_embeddings[k].tolist(),
                        "utterance": dialog[k][1],
                        "seq_num": k+1,
                        "domains": batch_domains[j],
                        "source_dataset": dataset_name,
                        "source_dialog_id": batch_ids[j],
                        "source_speaker": dialog[k][0]
                    }
                    if k > 0:
                        source["context_embedding"] = np.mean(dialog_embeddings[:k], axis=0).tolist()
                        
                    action = {
                        "_op_type": "index",
                        "_id": utterance_id,
                        "_source": source
                    } 
                    es_docs.append(action)
                    
            bulk(self.es, es_docs, index=self.es_index_name, chunk_size=self.es_chunk_size)
            
    def _get_scoring_query(self, context_embedding, domains, source_datasets, source_speakers):
        filter_clause = []
        if context_embedding is not None:
            filter_clause.append({
                "exists": {
                  "field": "context_embedding"
                }
            })
        else:
            filter_clause.append({
                "term": {
                    "seq_num": 1
                }
            })
        if domains is not None:
            filter_clause.append({
                "terms": {
                    "domains": domains
                }
            })
        if source_datasets is not None:
            filter_clause.append({
                "terms": {
                    "source_dataset": source_datasets
                }
            })
        if source_speakers is not None:
            filter_clause.append({
                "terms": {
                    "source_speaker": source_speakers
                }
            })
        
        if context_embedding is not None:
            score_func = "dotProduct" if self.normalize_embeddings else "cosineSimilarity"
            query = {
                "script_score": {
                    "query": {
                        "bool": {
                            "filter": filter_clause
                        }
                    },
                    "script": {
                        "source": "%s(params.query_vector, 'context_embedding') + 1.0" % score_func,
                        "params": {"query_vector": context_embedding.tolist()}
                    }
                }
            }
        else:
            query = {
                "function_score": {
                    "query": {
                        "bool": {
                            "filter": filter_clause
                        }
                    },
                    "random_score": {
                        "seed": 42,
                        "field": "utterance_id.keyword"
                    },
                    "boost_mode": "replace"
                }
            }
            
        query = {
            "_source": ["utterance_embedding", "utterance"],
            "query": query
        }
          
        return query