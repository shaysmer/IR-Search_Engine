from __future__ import annotations

import math
import pickle
import re
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
from google.cloud import storage


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


class GCSMultiFileReader:
    """
    Reads byte ranges from multiple GCS objects given locs=[(file_name, offset), ...].
    Handles cases where file_name already includes the base_dir prefix.
    """

    def __init__(self, bucket_name: str, base_dir: str):
        self.bucket_name = bucket_name
        self.base_dir = base_dir.strip("/")

        self._client = storage.Client()
        self._bucket = self._client.bucket(bucket_name)

    def _normalize_blob_name(self, f_name: str) -> str:
        f_name = f_name.lstrip("/")
        if self.base_dir == "":
            return f_name
        # If already starts with base_dir, don't double-prepend
        if f_name.startswith(self.base_dir + "/") or f_name == self.base_dir:
            return f_name
        return f"{self.base_dir}/{f_name}"

    def read(self, locs: Iterable[Tuple[str, int]], n_bytes: int) -> bytes:
        chunks = []
        remaining = n_bytes

        for f_name, offset in locs:
            if remaining <= 0:
                break

            blob_name = self._normalize_blob_name(str(f_name))
            blob = self._bucket.blob(blob_name)

            # staff logic: n_read = min(remaining, BLOCK_SIZE - offset)
            n_read = min(remaining, BLOCK_SIZE - int(offset))
            if n_read <= 0:
                continue

            # download byte range [start, end] inclusive
            start = int(offset)
            end = start + n_read - 1
            data = blob.download_as_bytes(start=start, end=end)

            chunks.append(data)
            remaining -= len(data)

        return b"".join(chunks)


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

    def __init__(
        self,
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

        # Load assets once (startup)
        self.doc_title_dic = self._load_pickle_from_gcs(self.title_dict_pkl_path)
        self.index_body = self._load_pickle_from_gcs(self.body_index_pkl_path)

        # sanity checks
        if not hasattr(self.index_body, "df") or not hasattr(self.index_body, "posting_locs"):
            raise ValueError("Loaded body index object doesn't look like an inverted index (df/posting_locs missing).")
        if not hasattr(self.index_body, "N"):
            raise ValueError("Loaded body index is missing attribute N (needed for IDF).")
        if not hasattr(self.index_body, "weights"):
            raise ValueError("Loaded body index is missing attribute weights (needed for doc_len/doc_norm).")

        self._reader = GCSMultiFileReader(bucket_name=self.bucket_name, base_dir=self.body_bins_folder)

    # ------------------------- Public API -------------------------

    def search(self, query: str, top_k: int = 100) -> List[Tuple[int, str]]:
        if not query:
            return []

        tokens = self._tokenize(query)
        if not tokens:
            return []

        query_tfidf = self._calc_query_tfidf(tokens, self.index_body)
        if not query_tfidf:
            return []

        doc_numerators = self._get_candidate_docs_body(tokens, query_tfidf)
        if not doc_numerators:
            return []

        scores = self._cosine_scores(doc_numerators, query_tfidf)
        if not scores:
            return []

        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        res: List[Tuple[int, str]] = []
        for doc_id, _score in top:
            title = self.doc_title_dic.get(doc_id)
            if title is not None:
                res.append((int(doc_id), title))
        return res

    # ------------------------- Tokenization -------------------------

    def _tokenize(self, text: str) -> List[str]:
        tokens = [m.group().lower() for m in RE_WORD.finditer(text)]
        return [t for t in tokens if t not in ALL_STOPWORDS]

    # ------------------------- TF-IDF helpers -------------------------

    def _calc_query_tfidf(self, query_tokens: List[str], index) -> Dict[str, float]:
        """
        tfidf(q, t) = (tf / |q|) * log(N / df(t))
        """
        q_tf = Counter(query_tokens)
        q_len = len(query_tokens)
        if q_len == 0:
            return {}

        N = index.N
        q_weights: Dict[str, float] = {}

        for term in np.unique(query_tokens):
            df = index.df.get(term)
            if not df:
                continue
            q_weights[term] = (q_tf[term] / q_len) * math.log(N / df)

        return q_weights

    @lru_cache(maxsize=25000)
    def _read_posting_list(self, term: str) -> List[Tuple[int, int]]:
        """
        Reads posting list from GCS bins folder (self.body_bins_folder).
        Each posting entry encoded as 6 bytes: 4 bytes doc_id, 2 bytes tf.
        """
        index = self.index_body

        if term not in index.posting_locs:
            return []

        df = index.df.get(term, 0)
        if df == 0:
            return []

        locs = index.posting_locs[term]
        b = self._reader.read(locs, df * TUPLE_SIZE)

        posting_list: List[Tuple[int, int]] = []
        for i in range(df):
            start = i * TUPLE_SIZE
            doc_id = int.from_bytes(b[start:start + 4], "big")
            tf = int.from_bytes(b[start + 4:start + 6], "big")
            posting_list.append((doc_id, tf))

        return posting_list

    def _get_candidate_docs_body(self, query_tokens: List[str], query_tfidf: Dict[str, float]) -> Dict[int, float]:
        """
        Builds dot-product numerator for cosine similarity:
            numerator(doc) += (tf/doc_len) * idf * q_tfidf(term)

        Assumes index.weights[doc_id][0] = doc_len
                index.weights[doc_id][1] = doc_norm_sq   (or similar)
        """
        index = self.index_body
        N = index.N

        numerators: Dict[int, float] = defaultdict(float)

        for term in np.unique(query_tokens):
            df = index.df.get(term)
            if not df:
                continue

            q_w = query_tfidf.get(term)
            if q_w is None:
                continue

            idf = math.log(N / df)
            posting_list = self._read_posting_list(term)
            if not posting_list:
                continue

            for doc_id, tf in posting_list:
                try:
                    doc_len = index.weights[doc_id][0]
                except Exception:
                    continue
                if not doc_len:
                    continue

                numerators[doc_id] += (tf / doc_len) * idf * q_w

        return numerators

    def _cosine_scores(self, numerators: Dict[int, float], query_tfidf: Dict[str, float]) -> Dict[int, float]:
        """
        cosine(doc, q) = numerator / (||doc|| * ||q||)
        where doc norm squared is assumed available at index.weights[doc_id][1]
        """
        index = self.index_body

        q_norm_sq = sum(w * w for w in query_tfidf.values())
        if q_norm_sq == 0:
            return {}
        q_norm = math.sqrt(q_norm_sq)

        scores: Dict[int, float] = {}
        for doc_id, num in numerators.items():
            try:
                doc_norm_sq = index.weights[doc_id][1]
            except Exception:
                continue
            if not doc_norm_sq:
                continue

            denom = math.sqrt(doc_norm_sq) * q_norm
            if denom == 0:
                continue

            scores[doc_id] = num / denom

        return scores

    # ------------------------- GCS helpers -------------------------

    def _load_pickle_from_gcs(self, blob_path: str):
        bucket = self._storage_client.bucket(self.bucket_name)
        blob = bucket.blob(blob_path)
        data = blob.download_as_bytes()
        return pickle.loads(data)
