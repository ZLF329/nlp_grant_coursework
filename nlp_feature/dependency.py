import spacy
import numpy as np
import json
from pathlib import Path


def get_sentence_depth_metrics(sent):
    """
    calculate the max and average dependency parse tree depth for a given sentence.
    """
    depths = []
    
    for token in sent:
        # skip punctuation tokens
        if token.is_punct:
            continue
            
        depth = 0
        current = token
        
        
        while current.head != current:
            depth += 1
            current = current.head
        
        depths.append(depth)
    
    if not depths:
        return 0, 0

    max_depth = max(depths)
    avg_depth = sum(depths) / len(depths)
    
    return max_depth, avg_depth


def analyze_single_file(sections):
    try:
        parser = spacy.load("en_core_web_sm")
    except OSError:
        print("loading en_core_web_sm ...")
        from spacy.cli import download
        download("en_core_web_sm")
        parser = spacy.load("en_core_web_sm")

    
    file_max_depths = []
    file_avg_depths = []
    
    for content in sections.values():
        if not content: 
            continue

        doc = parser(str(content))
        
        for sent in doc.sents:

            max_d, avg_d = get_sentence_depth_metrics(sent)
            file_max_depths.append(max_d)
            file_avg_depths.append(avg_d)
    
    if file_max_depths:
        final_max = max(file_max_depths)
        final_avg = sum(file_avg_depths) / len(file_avg_depths)
    else:
        final_max = 0
        final_avg = 0   
        
    return final_max, final_avg
