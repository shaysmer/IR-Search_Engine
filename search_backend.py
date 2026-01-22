from __future__ import annotations

import math
import pickle
import re
from collections import Counter, defaultdict

from typing import Dict, List, Tuple
from google.cloud import storage


# =============================================================================
# Stopwords + Tokenization
# =============================================================================

# --- Minimal embedded English stopwords set (based on common NLTK list) ---
# This is included directly in the code to avoid runtime dependency on:
#   nltk.download("stopwords")
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

# Extra stopwords typically appearing in Wikipedia-like corpora
# (common section words / boilerplate tokens)
CORPUS_STOPWORDS = frozenset([
    "category", "references", "also", "external", "links", "may", "first", "see", "history",
    "people", "one", "two", "part", "thumb", "including", "second", "following", "many", "however",
    "would", "became"
])
# Unified stopwords set used by the tokenizer
ALL_STOPWORDS = ENGLISH_STOPWORDS.union(CORPUS_STOPWORDS)

# Tokenization regex:
# - Accepts #, @, word chars
# - Allows internal apostrophe/hyphen
# - Token length roughly constrained to reduce noise
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

# =============================================================================
# Constants (not necessarily used by this file directly)
# =============================================================================

# Typical block size and tuple size used in posting file formats in IR assignments.
# They are currently not used inside this code, but kept for compatibility / reference.
BLOCK_SIZE = 1999998
TUPLE_SIZE = 6


# =============================================================================
# GCS Paths / Config
# =============================================================================
bucket_name = "shay-208886382-bucket"
file_path = "postings_gcp/index.pkl"

def load_index_from_gcp(bucket_name: str, file_path: str):
    """
       Download and deserialize (pickle) an inverted index object from GCS.

       Parameters
       ----------
       bucket_name : str
           Google Cloud Storage bucket name.
       file_path : str
           Object path inside the bucket (e.g., "postings_gcp/index.pkl").

       Returns
       -------
       inverted : object
           A deserialized inverted index instance. Expected to expose at least:
             - df (document frequency dict)
             - posting_locs (term -> list of (bin_filename, offset) or similar)
             - read_a_posting_list(bins_folder, term, bucket_name)
       """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    # Download raw bytes and unpickle into an index object
    contents = blob.download_as_bytes()
    inverted = pickle.loads(contents)
    return inverted

def load_title_dict_from_gcp(bucket_name: str, file_path: str):
    """
        Download and deserialize the doc_id -> title dictionary from GCS.

        Parameters
        ----------
        bucket_name : str
            Google Cloud Storage bucket name.
        file_path : str
            Object path inside the bucket (e.g., "doc_title_dic.pickle").

        Returns
        -------
        doc_title_dic : Dict[int, str]
            Mapping from wiki_id (document id) to its title string.
        """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    contents = blob.download_as_bytes()
    doc_title_dic = pickle.loads(contents)
    return doc_title_dic



class SearchEngine:
    """
        A minimal search backend implementing TF-IDF + cosine similarity over the BODY index.

        High-level flow:
          1) Tokenize query (lowercase, regex, remove stopwords)
          2) Compute query TF-IDF weights:
               w_q(term) = (tf_q(term) / |q|) * idf(term)
               idf(term) = log((N+1)/(df+1))   (smoothed)
          3) For each query term:
               - load posting list: [(doc_id, tf_d(term)), ...]
               - accumulate dot product:
                     score[doc] += w_q(term) * (tf_d(term) * idf(term))
          4) Normalize by query norm (cosine normalization partial):
               final_score = dot(doc, query) / ||query||
             Note: this code does NOT divide by ||doc||, so it is not a full cosine similarity.
             (If your assignment requires full cosine, you'd need precomputed doc norms.)

        Expected GCS objects:
          - body_index_pkl_path: "postings_gcp/index.pkl"
          - posting bin files located under: "postings_gcp/*.bin"
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

    # =============================================================================
    # Public API
    # =============================================================================

    def search(self, query, top_k=100):
        """
                Search for a query string over the BODY index using TF-IDF weighted scoring.

                Parameters
                ----------
                query : str
                    Raw query string provided by the user.
                top_k : int
                    Maximum number of results to return.

                Returns
                -------
                List[Tuple[str, str]]
                    A list of (wiki_id_as_str, title) pairs ordered by score descending.
                """
        # Tokenize query; if empty after stopword removal, return no results
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # Compute query tf-idf weights and term idf values (term -> (w_q, idf))
        q_data = self._calc_query_tfidf(query_tokens, self.index_body)
        if not q_data:
            return []

        # query norm
        q_norm = math.sqrt(sum(w_q * w_q for (w_q, _) in q_data.values()))
        if q_norm == 0:
            return []

        # Accumulate dot-product scores per document
        scores = defaultdict(float)

        for term, (w_q, idf) in q_data.items():
            # Load posting list for this term: [(doc_id, tf), ...]
            posting = self._posting_list(term)
            if not posting:
                continue

            # For each doc: w_d = tf * idf (standard tf-idf for document)
            # Add to dot product with query vector
            for doc_id, tf in posting:
                w_d = tf * idf
                scores[doc_id] += w_q * w_d

        if not scores:
            return []

        # Rank by normalized score (normalize only by query norm, not document norm)
        ranked = sorted(
            ((doc_id, score / q_norm) for doc_id, score in scores.items()),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        return self.find_title_doc_id(ranked)

    def find_title_doc_id(self, candidates):
        """
                Convert ranked candidates from (doc_id, score) into output tuples (doc_id, title).

                Parameters
                ----------
                candidates : Iterable[Tuple[int, float]]
                    Ranked candidates, each is (doc_id, score).

                Returns
                -------
                List[Tuple[str, str]]
                    (doc_id_as_str, title) for each candidate.
                """
        res = []
        for i, _ in candidates:
            title = self.doc_title_dic.get(i)
            if title is None:
                title = ""
            res.append((str(i), title))
        return res

    # =============================================================================
    # TF-IDF ֿ
    # =============================================================================
    def _calc_query_tfidf(self, query_tokens: List[str], index) -> Dict[str, Tuple[float, float]]:
        """
        Compute query-side TF-IDF weights.

        Returns a dict:
          term -> (w_q, idf)

        Definitions
        -----------
        - tf_q(term) = frequency of term in query
        - |q| = query length in tokens
        - df(term) = number of documents containing term (from index.df)
        - N = corpus size (#documents)

        Weighting:
          idf(term) = log((N + 1) / (df + 1))        # smoothed IDF
          w_q(term) = (tf_q(term) / |q|) * idf(term) # normalized query TF
        """
        q_tf = Counter(query_tokens)
        q_len = len(query_tokens)
        if q_len == 0:
            return {}

        # Corpus size: using title dictionary length as number of documents
        N = len(self.doc_title_dic)   # corpus size
        q_data = {}

        for term, tf in q_tf.items():
            # Document frequency for term
            df = index.df.get(term, 0)
            if df <= 0:
                continue

            # Smoothed IDF to avoid division by zero
            idf = math.log((N + 1) / (df + 1))
            # Query TF normalized by query length
            w_q = (tf / q_len) * idf
            q_data[term] = (w_q, idf)

        return q_data

    def _tokenize(self, text: str):
        """
               Tokenize a text into terms using a regex and remove stopwords.

               Steps:
                 1) Lowercase
                 2) Regex finditer with RE_WORD
                 3) Remove stopwords

               Parameters
               ----------
               text : str
                   Input string.

               Returns
               -------
               List[str]
                   Token list after filtering stopwords.
               """
        tokens = [m.group() for m in RE_WORD.finditer(text.lower())]
        return [t for t in tokens if t not in ALL_STOPWORDS]

    def _posting_list(self, term: str):
        """
                Load a posting list for a term from the inverted index.

                This method attempts to call the index's `read_a_posting_list` using
                (bins_folder, term, bucket_name). If it returns empty, it attempts a
                fallback: detect whether posting_locs store full paths already, and
                adjust folder argument accordingly.

                Parameters
                ----------
                term : str
                    A tokenized term.

                Returns
                -------
                List[Tuple[int, int]] or similar
                    Posting list typically of the form [(doc_id, tf), ...].
                    If nothing found or term not indexed, returns [].
                """
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
