# TF-IDF on the Cranfield Dataset

A compact, end-to-end **Information Retrieval** project that builds a TF-IDF vector space model over the **Cranfield collection**, ranks documents for user queries with **cosine similarity**, and evaluates results against ground-truth relevance judgments.

---

## 🚀 Features

- **Cranfield dataset included** (`cran.1400`, `cran.qry`, `cranqrel`) for immediate use.
- Clean **preprocessing**: tokenization, lowercasing, stopword removal, optional stemming/lemmatization.
- **TF, IDF, TF-IDF** computation with a sparse term–document matrix.
- **Cosine similarity** ranking for queries → top-k document retrieval.
- **Evaluation** via classic IR metrics (Precision@k, Recall@k, MAP; optionally NDCG/MRR).
- **Reproducible notebooks** for exploration and reporting.

---

## 🧱 Repository Structure

TF-IDF-cranDataSet/
├─ cran.1400 # Cranfield documents
├─ cran.qry # Cranfield queries
├─ cranqrel # Relevance judgments (qrels)
├─ cranqrel.readme # Notes/format description for qrels
├─ TF-IDF_Evaluation.ipynb # Main end-to-end notebook (index + retrieval + eval)
├─ TestOutOfData.ipynb # Additional experiments / out-of-sample tests
├─ Evaluation-1.py # Script version of evaluation pipeline
└─ .ipynb_checkpoints/ # Notebook checkpoints


> If you extend the project, consider adding modules like `preprocess.py`, `index.py`, `rank.py`, `eval.py` for a cleaner CLI.

---

## 🛠 Tech Stack

| Area            | Tools / Libraries (suggested)                           |
|-----------------|---------------------------------------------------------|
| Language        | Python 3.9+                                             |
| NLP / IR        | `scikit-learn` (TfidfVectorizer) or custom NumPy/SciPy  |
| Text Processing | `nltk` / `spacy` (stopwords, tokenizer, stem/lemmatize) |
| Math / Sparse   | `numpy`, `scipy`                                        |
| Notebooks       | `jupyter`, `pandas`, `matplotlib`                       |

> You can swap in `spacy` for robust tokenization/lemmatization, or keep it light with `nltk`.

---

## ⚡ Quickstart

### 1) Clone

git clone https://github.com/zakarich/TF-IDF-cranDataSet.git
cd TF-IDF-cranDataSet


### 2) (Optional) Create a virtual environment

python -m venv .venv

Windows
.venv\Scripts\activate

macOS / Linux
source .venv/bin/activate


### 3) Install dependencies

pip install -U pip
pip install numpy scipy scikit-learn nltk pandas matplotlib jupyter

Optional:
pip install spacy && python -m spacy download en_core_web_sm


### 4) Run the pipeline (script)

End-to-end evaluation run (adjust flags/paths inside the script if needed)
python Evaluation-1.py


### 5) Or use the notebooks

jupyter notebook

Open: TF-IDF_Evaluation.ipynb
(Optionally) TestOutOfData.ipynb for additional experiments


---

## 🧪 What the Pipeline Does

Load Cranfield documents/queries and qrels

Preprocess text (tokenize, normalize, remove stopwords, stem/lemmatize)

Build TF-IDF vectors for all documents

Convert queries to TF-IDF vectors using the same vocabulary

Rank documents per query with cosine similarity

Evaluate ranked lists with Precision@k, Recall@k, MAP (and optionally NDCG/MRR)

Export or display metrics and top-k results


---

## 🧩 Configuration Tips

Common toggles you may add inside the code/notebooks:
MIN_DF / MAX_DF thresholds for vocabulary pruning

Use sublinear TF scaling (log(1 + tf))

Normalize vectors (L2) before cosine similarity

Stemming vs Lemmatization (SnowballStemmer vs spaCy lemmatizer)

Stopword list source (nltk, spaCy, or custom domain list)

Top-k cutoff for evaluation (e.g., k=10, 20, 100)


---

## 📈 Example: Minimal TF-IDF with scikit-learn (snippet)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

docs = [...]   # list of document strings loaded from cran.1400
queries = [...]  # list of query strings loaded from cran.qry

tfidf = TfidfVectorizer(lowercase=True, stop_words='english', min_df=2)
X = tfidf.fit_transform(docs)      # (n_docs x n_terms)
Q = tfidf.transform(queries)       # (n_queries x n_terms)

scores = cosine_similarity(Q, X)   # (n_queries x n_docs)
ranked = scores.argsort(axis=1)[:, ::-1]  # rank docs per query (desc)
(In this repo, use the provided script/notebooks instead of this toy demo.)
```

## 🧪 Testing

# If you add pytest/CI later:
pytest -q
Current project focuses on experimentation via notebooks and the Evaluation-1.py script. For production-style testing, consider refactoring into modules and adding pytest suites.

## 🤝 Contribution

Contributions are welcome!

Ideas:
- Add BM25 / PL2 / QL (LM with Dirichlet/JM) baselines
- Add embedding-based rerankers (S-BERT) or LSA/LDA comparisons
- Extend evaluation to NDCG@k, MRR, MAP@k; plot PR and ROC curves
- Add a proper CLI: preprocess → index → search → eval with argparse
- Add caching (serialized vocab + TF-IDF matrix) for faster iterations
- Package as a small library (setup.cfg/pyproject.toml)
Workflow:
1) Fork the repo
2) Create a feature branch
3) Commit with clear messages
4) Open a PR with notes and (if possible) a notebook demo

## 📬 Contact

Maintainer: @zakarich  (https://github.com/zakarich)
Open an Issue for bugs/ideas, or connect for collaboration.
📄 License

MIT License

Copyright (c) 2025 zak

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
## ⭐ Citation (optional)
If you use or extend this repo in academic work, please cite the Cranfield collection and this repository.

@misc{zakarich_tfidf_cranfield,
  title  = {TF-IDF on the Cranfield Dataset},
  author = {Zakariyae Chatri},
  year   = {2025},
  url    = {https://github.com/zakarich/TF-IDF-cranDataSet}
}
