**# Hybrid Search RAG Assistant

A Retrieval-Augmented Generation (RAG) application that answers questions from uploaded documents using hybrid keyword and semantic search, Reciprocal Rank Fusion (RRF), and a locally hosted language model.

## Features

- Upload PDF and TXT documents
- Extract text and split it into overlapping chunks
- Retrieve relevant passages using BM25 keyword search
- Perform semantic retrieval using Sentence Transformers and vector similarity
- Combine search rankings using Reciprocal Rank Fusion
- Generate evidence-grounded answers using Ollama
- Display retrieved evidence and source filenames
- Evaluate retrieval, answer quality, refusal behavior, and response latency
- Export evaluation results to CSV

## Tech Stack

- Python
- Streamlit
- Sentence Transformers
- BM25 (`rank_bm25`)
- NumPy
- Ollama
- Qwen3 8B
- PyPDF

## Architecture

1. **Document ingestion:** Load PDF and TXT files and extract their text.
2. **Chunking:** Split documents into overlapping text passages.
3. **Indexing:** Create BM25 and semantic vector indexes.
4. **Hybrid retrieval:** Retrieve keyword and semantic matches, then combine their rankings using RRF.
5. **Answer generation:** Pass retrieved evidence to the local language model.
6. **Evaluation:** Measure source retrieval, assess generated answers, check refusal behavior, and record latency.

## Project Structure

```text
hybrid-rag-assistant/
├── app.py
├── evaluate.py
├── evaluation_questions.json
├── evaluation_results.csv
├── data/
│   ├── leave_policy.txt
│   ├── it_support.txt
│   └── expense_policy.txt
└── src/
    ├── __init__.py
    └── retrieval.py
```

## Run Locally

Install the dependencies in your Python environment:

```bash
pip install streamlit sentence-transformers rank-bm25 ollama pypdf numpy
```

Install Ollama separately, start its local service, and download the model:

```bash
ollama pull qwen3:8b
```

Run the application:

```bash
streamlit run app.py
```

Run the evaluation suite:

```bash
python evaluate.py
```

The evaluation script reads TXT documents from `data/` and test cases from `evaluation_questions.json`. It saves detailed results to `evaluation_results.csv`.

## Initial Evaluation

Tested against eight questions using three sample policy documents.

- Source hit rate: 6/6 (100%)
- Answer correctness: 6/6 (100%), according to an automated LLM judge
- Refusal phrase detection: 2/2 (100%)
- Average response latency: approximately 15.6 seconds

These are initial results from a small test set, not a guarantee of general accuracy. The answer judge uses the same language model as the answer generator, and the results require further validation.

## Future Improvements

- Expand the evaluation dataset with more answerable and unanswerable questions
- Improve retrieval ranking and measure Recall@k and MRR
- Reduce response latency
- Add more robust citation and grounding checks
- Compare different embedding and language models
- Add automated regression tests and deployment documentation

## Limitations

Performance depends on document quality, retrieval relevance, model behavior, and available hardware. Generated answers and automated evaluation scores should be reviewed before relying on them for important decisions.
**
