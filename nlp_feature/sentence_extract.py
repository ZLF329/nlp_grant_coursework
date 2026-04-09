import json
import os
import spacy
import re
# default structure in json file
default_structure = {
    "APPLICATION DETAILS": [
        "Scientific Abstract",
        "Plain English Summary",
        "Changes from Previous Stage",
        "Working with People and Communities Summary"
    ]
}


def load_json(path: str, filename: str) -> dict:
    '''
    Read json data from given path and filename
    
    Args:
        path (str): Directory path containing the JSON file
        filename (str): Name of the JSON file

    Returns:
        dict: Parsed JSON content
    '''
    file_path = os.path.join(path, filename)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data

def get_sections(data: dict, default_structure: dict) -> dict:
    '''
    Extract specified section texts from JSON data according to a predefined structure.

    Args:
        data (dict): Parsed JSON data loaded from an application file.
        default_structure (dict): A mapping that defines which section titles

    Returns:
        dict: A flat dictionary mapping section titles to their extracted text.
    '''
    sections = {}
    for block, titles in default_structure.items():
        block_data = data.get(block, {})
        for title in titles:
            sections[title] = block_data.get(title)

    return sections

def merge_sections(sections: dict, default_structure: dict) -> str:
    '''Merge the "APPLICATION DETAILS" part into a whole sentence'''
    whole_section = []
    # merge "APPLICATION DETAILS" part initially
    titles = default_structure.get("APPLICATION DETAILS", [])
    if not titles:
        return ""
    for title in titles:
        txt = sections.get(title)
        if txt:
            txt = txt.strip()
            if txt:
                whole_section.append(txt)
    return " ".join(whole_section)

def build_sentence_segment(model: str="en_core_web_sm"):
    """ 
    Build and return a spaCy NLP object for sentence segmentation only. 

    Args: 
        model (str): spaCy model name. 
        
    Returns: 
        spacy.language.Language: Loaded spaCy pipeline configured for sentence splitting. 
    """
    nlp = spacy.load(
        model,
        disable=["tagger", "parser", "ner", "lemmatizer", "attribute_ruler"]
    )
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    return nlp

def split_text_into_sentences(text: str, nlp) -> list[str]:
    """
    Split text into sentences using spaCy.

    Args:
        text (str): Input text to segment.
        nlp: spaCy Language object.
    Returns:
        List[str]: A list of cleaned sentence strings.
    """
    if not text:
        return []
    
    # normalize for sentence-level processing
    text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()

    doc = nlp(text)
    sents = []
    for s in doc.sents:
        sent = s.text.strip()
        if sent:
            sents.append(sent)

    return sents

def merge_and_split_sentence(sections: dict, default_structure: dict, nlp) -> list[str]:
    """
    Merge specified sections and split into sentences using spaCy.

    Args:
        sections (dict): Extracted section texts.
        default_structure (dict): Schema defining which sections to use.
        nlp: spaCy Language object.

    Returns:
        List[str]: Clean sentences.
    """
    whole_sentence = merge_sections(sections, default_structure)
    return split_text_into_sentences(whole_sentence, nlp)

def sentence_words_count(sentence: list[str]) -> list:
    '''Return the length of each sentence'''
    length = []
    for sent in sentence:
        length.append(len(sent.split()))

    return length

def save_features(data: dict, output_path: str, original_filename: str):
    '''Save json file'''
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    base_name = os.path.splitext(original_filename)[0]
    new_filename = f"{base_name}_features.json"
    full_path = os.path.join(output_path, new_filename)
    
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print(f"Json file save to: {full_path} successfully")