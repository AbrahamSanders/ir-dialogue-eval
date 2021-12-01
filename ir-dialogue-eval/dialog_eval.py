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
                 embedding_model="all-MiniLM-L12-v2", normalize_embeddings=True,
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
                        source["prior_utterance_embedding"] = dialog_embeddings[k-1].tolist()
                    if k > 1:
                        source["context_embedding"] = np.mean(dialog_embeddings[:(k-1)], axis=0).tolist()
                        
                    action = {
                        "_op_type": "index",
                        "_id": utterance_id,
                        "_source": source
                    } 
                    es_docs.append(action)
                    
            bulk(self.es, es_docs, index=self.es_index_name, chunk_size=self.es_chunk_size)
    
    def score_utterances(self, utterances_in_context, reference_utterances=None, domains=None, 
                         source_datasets=None, source_speakers=None, max_samples=100, context_weight=0.5):
        
        self._validate_score_params(utterances_in_context, reference_utterances, domains, 
                                    source_datasets, source_speakers)
        
        scores = []
        n_samples = max_samples-1 if reference_utterances else max_samples
        for i in trange(len(utterances_in_context), desc="Utterances", disable=not self.show_progress):
            utt_in_context = utterances_in_context[i]
            
            #Embed everything
            embeddings = self.embedding_model.encode(utt_in_context, batch_size=self.embed_batch_size, 
                                                     normalize_embeddings=self.normalize_embeddings, 
                                                     show_progress_bar=False)
            
            utterance_embedding = embeddings[-1]
            prior_utterance_embedding = embeddings[-2] if len(utt_in_context) > 1 else None
            context_embedding = np.mean(embeddings[:-2], axis=0) if len(utt_in_context) > 2 else None
            
            if reference_utterances:
                ref_utterance_embedding = self.embedding_model.encode(reference_utterances[i], 
                                                                      normalize_embeddings=self.normalize_embeddings, 
                                                                      show_progress_bar=False)
                
            #Get nearest neighbor contexts from Elasticsearch
            scoring_query = self._get_scoring_query(context_embedding, prior_utterance_embedding, domains, 
                                                    source_datasets, source_speakers, context_weight)
            results = self.es.search(scoring_query, self.es_index_name, size=n_samples)
            results = [{"_score": hit["_score"], **hit["_source"]} for hit in results["hits"]["hits"]]
            
            if len(utt_in_context) > 1:
                context_cosines = np.array([r["_score"] for r in results])-1.0 # undo +1.0 in ES script score
            else:
                context_cosines = np.ones(len(results), dtype=np.float64)
            nearest_utterances = [r["utterance"] for r in results]
            nearest_utterance_embeddings = np.vstack([r["utterance_embedding"] for r in results]).astype(np.float32)
            
            if reference_utterances:
                context_cosines = np.concatenate((context_cosines, [1.0]))
                nearest_utterances.append(reference_utterances[i])
                nearest_utterance_embeddings = np.vstack((nearest_utterance_embeddings, ref_utterance_embedding))
                
            #Compute max utterance similarity weighted by context similarity
            shifted_context_cosines = context_cosines + (1-np.max(context_cosines))
            utterance_cosines = 1 - np.squeeze(cdist(nearest_utterance_embeddings, 
                                                     np.expand_dims(utterance_embedding, 0), 
                                                     metric="cosine"), axis=1)
            weighted_cosines = utterance_cosines * shifted_context_cosines
            
            score = np.max(weighted_cosines)
            scores.append(score)
        
        return np.array(scores)
    
    def _get_scoring_query(self, context_embedding, prior_utterance_embedding, domains, 
                           source_datasets, source_speakers, context_weight):
        filter_clause = []
        if prior_utterance_embedding is not None:
            filter_clause.append({
                "exists": {
                  "field": "prior_utterance_embedding"
                }
            })
        else:
            filter_clause.append({
                "term": {
                    "seq_num": 1
                }
            })
        if context_embedding is not None:
            filter_clause.append({
                "exists": {
                  "field": "context_embedding"
                }
            })
        if domains:
            filter_clause.append({
                "terms": {
                    "domains": domains
                }
            })
        if source_datasets:
            filter_clause.append({
                "terms": {
                    "source_dataset": source_datasets
                }
            })
        if source_speakers:
            filter_clause.append({
                "terms": {
                    "source_speaker": source_speakers
                }
            })
        
        if prior_utterance_embedding is not None:
            score_func = "dotProduct" if self.normalize_embeddings else "cosineSimilarity"
            if context_embedding is not None and context_weight > 0:
                script_source = ("(({0} * {2}(params.context_embedding, 'context_embedding')) +"
                                 " ({1} * {2}(params.prior_utterance_embedding, 'prior_utterance_embedding'))) + 1.0")
                script_source = script_source.format(context_weight, 1-context_weight, score_func)
                script_params = {"context_embedding": context_embedding.tolist(), 
                                 "prior_utterance_embedding": prior_utterance_embedding.tolist()}
            else:
                script_source = "%s(params.prior_utterance_embedding, 'prior_utterance_embedding') + 1.0" % score_func
                script_params = {"prior_utterance_embedding": prior_utterance_embedding.tolist()}
                
            query = {
                "script_score": {
                    "query": {
                        "bool": {
                            "filter": filter_clause
                        }
                    },
                    "script": {
                        "source": script_source,
                        "params": script_params
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
    
    def _validate_score_params(self, utterances_in_context, reference_utterances, domains, 
                               source_datasets, source_speakers):
        if reference_utterances and len(reference_utterances) != len(utterances_in_context):
            raise ValueError("If provided, reference_utterances must be the same length as utterances_in_context.")
                
        if domains and len(domains) != len(utterances_in_context):
            raise ValueError("If provided, domains must be the same length as utterances_in_context.")
            
        if source_datasets and len(source_datasets) != len(utterances_in_context):
            raise ValueError("If provided, source_datasets must be the same length as utterances_in_context.")
            
        if source_speakers and len(source_speakers) != len(utterances_in_context):
            raise ValueError("If provided, source_speakers must be the same length as utterances_in_context.")