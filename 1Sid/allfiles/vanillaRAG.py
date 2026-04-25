# ============================
# 1. IMPORTS
# ============================
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import faiss
import numpy as np
import torch
import pandas as pd
import re
from collections import Counter

# ============================
# 2. CONFIG
# ============================
DATASETS = [
    ("squad", "train[:50]"),
    # ("natural_questions", "train[:200]"),
    # ("trivia_qa", "train[:200]")
]

MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.1",
    "meta-llama/Meta-Llama-3-8B-Instruct"
]

TOP_K = 3
WIKI_VARIANT = "no_embeddings"
WIKI_LIMIT = 50000

WIKI_SHARDS = {
    "dummy.no_embeddings": 1,
    "no_embeddings": 28,
}

# ============================
# 3. LOAD WIKI DPR
# ============================
def load_wiki_dpr(limit=WIKI_LIMIT, variant=WIKI_VARIANT):
    if variant not in WIKI_SHARDS:
        raise ValueError(
            f"Unsupported wiki_dpr variant: {variant}. "
            f"Choose one of {list(WIKI_SHARDS)}."
        )

    shard_count = WIKI_SHARDS[variant]
    data_files = {
        "train": [
            (
                "hf://datasets/facebook/wiki_dpr/"
                f"data/psgs_w100/{variant}/"
                f"train-{i:05d}-of-{shard_count:05d}.parquet"
            )
            for i in range(shard_count)
        ]
    }

    wiki_stream = load_dataset(
        "parquet",
        data_files=data_files,
        split="train",
        streaming=True
    )

    if limit is not None:
        wiki_stream = wiki_stream.take(limit)

    return list(wiki_stream)

print("Loading Wikipedia DPR...")
wiki = load_wiki_dpr()
documents = [doc["text"] for doc in wiki]

# ============================
# 4. RETRIEVER
# ============================
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embed_model.encode(documents, show_progress_bar=True)

index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(np.array(doc_embeddings))

def retrieve(query, k=TOP_K):
    q_emb = embed_model.encode([query])
    distances, indices = index.search(q_emb, k)

    docs = [documents[i] for i in indices[0]]
    scores = distances[0]

    probs = np.exp(-scores) / np.sum(np.exp(-scores))
    return docs, probs

# ============================
# 5. LOAD NLI MODEL
# ============================
print("Loading NLI model...")
nli_model = pipeline(
    "text-classification",
    model="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
    device=0 if torch.cuda.is_available() else -1
)

# ============================
# 6. GENERATION
# ============================
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto"
    )
    return tokenizer, model

def generate(tokenizer, model, query, context):
    prompt = f"""
    Answer ONLY using the context.
    If not found, say "I don't know".

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ============================
# 7. VANILLA RAG
# ============================
def rag(query, tokenizer, model):
    docs, probs = retrieve(query)

    answers = []
    for doc in docs:
        ans = generate(tokenizer, model, query, doc)
        answers.append(ans)

    best_idx = np.argmax(probs)
    final = answers[best_idx]

    return docs, final

# ============================
# 8. METRICS
# ============================
def normalize(text):
    return re.sub(r"[^\w\s]", "", text.lower())

def compute_em(pred, gt):
    return int(normalize(pred) == normalize(gt))

def compute_f1(pred, gt):
    pred_tokens = normalize(pred).split()
    gt_tokens = normalize(gt).split()

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)

    return 2 * precision * recall / (precision + recall)

# ============================
# 9. HALLUCINATION METRICS
# ============================
def compute_faithfulness(context, answer):
    if answer.strip() == "":
        return 0

    result = nli_model(
        f"{context} </s> {answer}",
        candidate_labels=["entailment", "contradiction", "neutral"]
    )

    label = result["labels"][0]
    score = result["scores"][0]

    if label == "entailment":
        return score
    elif label == "contradiction":
        return -score
    else:
        return 0

def grounding_score(answer, context):
    ans_tokens = set(normalize(answer).split())
    ctx_tokens = set(normalize(context).split())

    if len(ans_tokens) == 0:
        return 0

    return len(ans_tokens & ctx_tokens) / len(ans_tokens)

def hallucination_score(context, answer):
    faith = compute_faithfulness(context, answer)
    overlap = grounding_score(answer, context)

    return 0.7 * faith + 0.3 * overlap

def is_hallucinated(score, threshold=0.3):
    return score < threshold

# ============================
# 10. DATASET PARSER
# ============================
def parse_sample(sample, dataset_name):
    if dataset_name == "squad":
        q = sample["question"]
        gt = sample["answers"]["text"][0] if sample["answers"]["text"] else ""
    else:
        q = sample["question"]
        gt = ""

    return q, gt

# ============================
# 11. RUN EXPERIMENT
# ============================
all_results = []

for dataset_name, split in DATASETS:
    print(f"\nDataset: {dataset_name}")

    dataset = load_dataset(dataset_name, split=split)

    for model_name in MODELS:
        print(f"\nModel: {model_name}")

        tokenizer, model = load_model(model_name)

        for i, sample in enumerate(dataset):
            query, gt = parse_sample(sample, dataset_name)

            docs, pred = rag(query, tokenizer, model)
            context = " ".join(docs)

            em = compute_em(pred, gt)
            f1 = compute_f1(pred, gt)

            faith = compute_faithfulness(context, pred)
            overlap = grounding_score(pred, context)
            h_score = hallucination_score(context, pred)
            hallucinated = is_hallucinated(h_score)

            row = {
                "dataset": dataset_name,
                "model": model_name,
                "query": query,
                "ground_truth": gt,
                "prediction": pred,

                "EM": em,
                "F1": f1,

                "faithfulness": faith,
                "grounding": overlap,
                "hallucination_score": h_score,
                "is_hallucinated": hallucinated
            }

            all_results.append(row)

            print(f"{i} | EM:{em} F1:{f1} H:{hallucinated}")

# ============================
# 12. SAVE CSV
# ============================
df = pd.DataFrame(all_results)
df.to_csv("rag_hallucination_results.csv", index=False)

print("\nSaved to rag_hallucination_results.csv")

# ============================
# 13. SUMMARY
# ============================
summary = df.groupby(["dataset", "model"]).agg({
    "EM": "mean",
    "F1": "mean",
    "faithfulness": "mean",
    "hallucination_score": "mean",
    "is_hallucinated": "mean"
}).reset_index()

print("\nSUMMARY:")
print(summary)
