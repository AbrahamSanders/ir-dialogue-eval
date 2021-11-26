import numpy as np
import math
import itertools
from tqdm import trange
from elasticsearch.helpers import bulk
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

from index_setup import verify_and_configure_index

class DatasetImporter(object):
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
                    context_embedding = None if k == 0 else np.mean(dialog_embeddings[:k], axis=0).tolist()
                    utterance_embedding = dialog_embeddings[k].tolist()
                    
                    action = {
                        "_op_type": "index",
                        "_id": "%s_%s_%s" % (dataset_name, batch_ids[j], k+1),
                        "doc": {
                            "context_embedding": context_embedding,
                            "utterance_embedding": utterance_embedding,
                            "utterance": dialog[k][1],
                            "speaker": dialog[k][0],
                            "seq_num": k+1,
                            "domains": batch_domains[j],
                            "source_dataset": dataset_name,
                            "source_dialog_id": batch_ids[j]
                        }
                    }
                    es_docs.append(action)
                    
            bulk(self.es, es_docs, index=self.es_index_name, chunk_size=self.es_chunk_size)