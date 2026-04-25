import re
import pickle
import streamlit as st
from janome.tokenizer import Tokenizer
from config import TITLE_STOP_WORDS, TITLE_SEMANTIC_MAP, VECTOR_FILE, LOCAL_MODEL_PATH

tokenizer = Tokenizer()

def extract_and_map_title_words(title_raw):
    if not isinstance(title_raw, str):
        return []
    clean_title = re.sub(r'\[.*?\]', ' ', title_raw).strip()
    extracted_words = []
    for token in tokenizer.tokenize(clean_title):
        word = token.surface
        part_of_speech = token.part_of_speech.split(',')[0]
        if part_of_speech in ['名詞', '動詞', '形容詞'] and word not in TITLE_STOP_WORDS and len(word) >= 1:
            mapped_word = TITLE_SEMANTIC_MAP.get(word, word)
            extracted_words.append(mapped_word)
    return extracted_words

@st.cache_resource(show_spinner="正在将 AI 语义矩阵载入内存...")
def load_semantic_engine():
    from sentence_transformers import SentenceTransformer
    import torch
    model = SentenceTransformer(
        LOCAL_MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
    )
    with open(VECTOR_FILE, 'rb') as f:
        data = pickle.load(f)
        
    corpus_embeddings = torch.tensor(data['embeddings'])
    corpus_ids = [str(i) for i in data['ids']]
    
    # 建立字典映射实现 O(1) 查找
    id_to_index = {link_id: idx for idx, link_id in enumerate(corpus_ids)}
    
    return model, corpus_embeddings, corpus_ids, id_to_index
