def clean_up_tokenization(out_string):
    #Adapted from original method at: 
    #https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_utils_base.py
    out_string = (
        out_string.replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" ' ", "'")
        .replace(" n't", "n't")
        .replace(" 'm", "'m")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re")
        .strip()
    )
    return out_string