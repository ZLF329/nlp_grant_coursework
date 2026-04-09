import numpy as np
from sentence_transformers import SentenceTransformer, util

def get_coherence_stats(data_dict: dict, threshold: float = 0.1):
    
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    all_scores = []
    low_coherence_count = 0
    
    # Tracking variables
    min_content_avg = float('inf')
    worst_content = None
    
    min_pair_score = float('inf')
    worst_pair = (None, None)
    worst_title = None

    for key, content_body in data_dict.items():
        if not content_body: 
            continue
            
        text = str(content_body)
        # Split text into sentences and filter empty strings
        sentences = [s.strip() for s in text.strip().split('\n') if s.strip()]
        
        if len(sentences) < 2:
            continue

        # Generate embeddings
        embeddings = model.encode(sentences, convert_to_tensor=True)
        current_block_scores = []
        
        
        for i in range(len(sentences) - 1):
            # Compute cosine similarity
            sim = util.cos_sim(embeddings[i], embeddings[i+1])
            score = sim.item()
            
            current_block_scores.append(score)
            all_scores.append(score)
            
            # Check for low coherence count
            if score < threshold:
                low_coherence_count += 1
            
            # Update the absolute worst pair found so far
            if score < min_pair_score:
                min_pair_score = score
                worst_pair = (sentences[i], sentences[i+1])

        # Update the content block with the lowest average score
        if current_block_scores:
            avg_score_for_block = np.mean(current_block_scores)
            if avg_score_for_block < min_content_avg:
                min_content_avg = avg_score_for_block
                worst_content = content_body
                worst_title = key

    global_average_score = np.mean(all_scores) if all_scores else 0.0
    
    return global_average_score, low_coherence_count, worst_content, worst_title