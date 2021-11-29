import logging

def verify_and_configure_index(es, index_name, embedding_dims):
    if es.indices.exists(index_name):
        logging.info("Verifying existing index '%s'..." % index_name)
        index = es.indices.get(index_name)
        mapping_properties = index[index_name]["mappings"]["properties"]
        index_embedding_dims = mapping_properties["context_embedding"]["dims"]

        if embedding_dims != index_embedding_dims:
            raise ValueError("Dims of the specified embedding model (%d) do not match the dims of "
                             "the 'context_embedding' and 'utterance_embedding' fields (%d) "
                             "in index '%s'." 
                             % (embedding_dims, index_embedding_dims, index_name))     
    else:
        logging.info("Creating index '%s'..." % index_name)
        mappings = {
            "properties": {
                "utterance_id": {
                    "type": "text",
                    "fields": {
                        "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                        }
                    }
                },
                "context_embedding": {
                    "type": "dense_vector",
                    "dims": embedding_dims
                },
                "prior_utterance_embedding": {
                    "type": "dense_vector",
                    "dims": embedding_dims
                },
                "utterance_embedding": {
                    "type": "dense_vector",
                    "dims": embedding_dims
                },
                "utterance": {
                    "type": "text",
                    "fields": {
                        "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                        }
                    }
                },
                "seq_num": {
                    "type": "integer"
                },
                "domains": {
                    "type": "text",
                    "fields": {
                        "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                        }
                    }
                },
                "source_dataset": {
                    "type": "text",
                    "fields": {
                        "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                        }
                    }
                },
                "source_dialog_id": {
                    "type": "text",
                    "fields": {
                        "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                        }
                    }
                },
                "source_speaker": {
                    "type": "text",
                    "fields": {
                        "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                        }
                    }
                }
            }
        }
        es.indices.create(index_name, {
            "mappings": mappings
        })
