from __future__ import annotations

import math
import pickle
import re
from collections import Counter, defaultdict
from contextlib import closing
from functools import lru_cache
from typing import Dict, List, Tuple, Iterable, Optional
import numpy as np
from google.cloud import storage
from inverted_index_gcp import InvertedIndex



# --- Minimal (embedded) English stopwords set (based on common NLTK list) ---
# (No nltk.download needed)
ENGLISH_STOPWORDS = frozenset({
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours","yourself","yourselves",
    "he","him","his","himself","she","her","hers","herself","it","its","itself","they","them","their",
    "theirs","themselves","what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does","did","doing","a","an",
    "the","and","but","if","or","because","as","until","while","of","at","by","for","with","about",
    "against","between","into","through","during","before","after","above","below","to","from","up",
    "down","in","out","on","off","over","under","again","further","then","once","here","there","when",
    "where","why","how","all","any","both","each","few","more","most","other","some","such","no","nor",
    "not","only","own","same","so","than","too","very","s","t","can","will","just","don","should","now",
    "d","ll","m","o","re","ve","y","ain","aren","couldn","didn","doesn","hadn","hasn","haven","isn",
    "ma","mightn","mustn","needn","shan","shouldn","wasn","weren","won","wouldn"
})

CORPUS_STOPWORDS = frozenset([
    "category", "references", "also", "external", "links", "may", "first", "see", "history",
    "people", "one", "two", "part", "thumb", "including", "second", "following", "many", "however",
    "would", "became"
])

ALL_STOPWORDS = ENGLISH_STOPWORDS.union(CORPUS_STOPWORDS)

# Token pattern (similar spirit to staff solutions)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

# In staff code inverted_index_gcp.py:
# BLOCK_SIZE = 1999998, TUPLE_SIZE=6 (doc_id 4 bytes + tf 2 bytes)
BLOCK_SIZE = 1999998
TUPLE_SIZE = 6


bucket_name = "shay-208886382-bucket"
file_path = "postings_gcp/index.pkl"

def load_index_from_gcp(bucket_name: str, file_path: str):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    contents = blob.download_as_bytes()
    inverted = pickle.loads(contents)
    return inverted

# file_path = "doc_title_dic.pickle"
def load_title_dict_from_gcp(bucket_name: str, file_path: str):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    contents = blob.download_as_bytes()
    doc_title_dic = pickle.loads(contents)
    return doc_title_dic



class SearchEngine:
    """
    Minimal backend:
      - TF-IDF cosine similarity over BODY inverted index stored on GCS.
      - returns [(wiki_id, title), ...] up to top_k.

    Expected GCS objects (your current setup):
      - body_index_pkl_path: "postings_gcp/index.pkl"
      - posting bin files under: "postings_gcp/....bin"
      - title_dict_pkl_path: "doc_title_dic.pickle"
    """

    def __init__(self,
        bucket_name: str,
        body_index_pkl_path: str = "postings_gcp/index.pkl",
        body_bins_folder: str = "postings_gcp",
        title_dict_pkl_path: str = "doc_title_dic.pickle",
    ):
        self.bucket_name = bucket_name
        self.body_index_pkl_path = body_index_pkl_path
        self.body_bins_folder = body_bins_folder
        self.title_dict_pkl_path = title_dict_pkl_path

        self._storage_client = storage.Client()

        self.index_body = load_index_from_gcp(bucket_name, self.body_index_pkl_path)
        self.doc_title_dic = load_title_dict_from_gcp(bucket_name, self.title_dict_pkl_path)

    # ------------------------- Public API -------------------------
    def search(self, query, top_k=100):
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        q_data = self._calc_query_tfidf(query_tokens, self.index_body)
        if not q_data:
            return []

        # query norm
        q_norm = math.sqrt(sum(w_q * w_q for (w_q, _) in q_data.values()))
        if q_norm == 0:
            return []

        scores = defaultdict(float)

        for term, (w_q, idf) in q_data.items():
            posting = self._posting_list(term)  # [(doc_id, tf), ...]
            if not posting:
                continue

            for doc_id, tf in posting:
                w_d = tf * idf
                scores[doc_id] += w_q * w_d

        if not scores:
            return []

        ranked = sorted(
            ((doc_id, score / q_norm) for doc_id, score in scores.items()),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        return self.find_title_doc_id(ranked)

    def find_title_doc_id(self, candidates):
        res = []
        for i, _ in candidates:
            title = self.doc_title_dic.get(i)
            if title is None:
                title = ""  # שלא יחזור None
            res.append((str(i), title))
        return res


    # ------------------------- TF-IDF helpers -------------------------

    def _calc_query_tfidf(self, query_tokens: List[str], index) -> Dict[str, Tuple[float, float]]:
        """
        Returns dict:
          term -> (w_q, idf)

        w_q = (tf(term in query)/|q|) * idf
        idf = log((N+1)/(df+1))
        """
        q_tf = Counter(query_tokens)
        q_len = len(query_tokens)
        if q_len == 0:
            return {}

        N = len(self.doc_title_dic) or 1  # corpus size
        q_data = {}

        for term, tf in q_tf.items():
            df = index.df.get(term, 0)
            if df <= 0:
                continue

            idf = math.log((N + 1) / (df + 1))  # smoothing
            w_q = (tf / q_len) * idf
            q_data[term] = (w_q, idf)

        return q_data

    def _tokenize(self, text: str):
        tokens = [m.group() for m in RE_WORD.finditer(text.lower())]
        return [t for t in tokens if t not in ALL_STOPWORDS]

    def _posting_list(self, term: str):
        pl = self.index_body.read_a_posting_list(self.body_bins_folder, term, self.bucket_name)
        if pl:
            return pl

        locs = getattr(self.index_body, "posting_locs", {}).get(term, [])
        if not locs:
            return []

        first_fname = locs[0][0]
        if isinstance(first_fname, str) and first_fname.startswith(self.body_bins_folder.rstrip("/") + "/"):
            return self.index_body.read_a_posting_list("", term, self.bucket_name)

        return []
