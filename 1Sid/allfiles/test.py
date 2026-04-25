

# ============================
# 1. IMPORTS
# ============================
import os
import gc
import pickle
import numpy as np
import pandas as pd
import torch
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from collections import Counter

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# ============================
# 2. CONFIG
# ============================
#for dataset_name, config, split, label in DATASETS:

DATASETS = [
    ("allenai/scifact", "claims", "validation[:10]", "SciFact")
]

MODELS = [
    ("unsloth/Qwen2.5-7B-Instruct-bnb-4bit", "local"),
    ("unsloth/mistral-7b-instruct-v0.3-bnb-4bit", "local")
]

TOP_K = 3

INDEX_PATH = "scifact.index"
DOCS_PATH  = "scifact_texts.pkl"

# ============================
# 3. LOAD RETRIEVER
# ============================
print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda:0")

if os.path.exists(INDEX_PATH):
    print("Loading cached FAISS index...")
    index = faiss.read_index(INDEX_PATH)
    with open(DOCS_PATH, "rb") as f:
        docs_db, doc_ids = pickle.load(f)
else:
    print("Building SciFact corpus...")
    corpus = load_dataset("allenai/scifact", "corpus", split="train")

    docs_db = []
    doc_ids = []
    
    for doc in corpus:
        text = doc["title"] + ". " + " ".join(doc["abstract"])
        docs_db.append(text)
        doc_ids.append(doc["doc_id"])

    embeddings = embed_model.encode(
        docs_db, batch_size=256, show_progress_bar=True
    ).astype("float32")

    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump((docs_db, doc_ids), f)
        

# move to CPU
embed_model.to("cpu")
torch.cuda.empty_cache()
gc.collect()

def retrieve(query, k=TOP_K):
    q_emb = embed_model.encode([query], device="cpu").astype("float32")

    faiss.normalize_L2(q_emb)   # ← ADD THIS LINE HERE

    distances, indices = index.search(q_emb, k)
    
    docs = [docs_db[i] for i in indices[0]]
    ids  = [doc_ids[i] for i in indices[0]]

    scores = distances[0]
    probs = np.exp(scores - np.max(scores)) / np.sum(np.exp(scores - np.max(scores)))

    return docs, ids, probs
# ============================
# 4. NLI MODEL
# ============================
nli_model = pipeline(
    "text-classification",
    model="MoritzLaurer/deberta-v3-base-zeroshot-v2",
    device=0
)

# ============================
# 5. GENERATOR
# ============================
def load_model(name):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        name,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model.eval()
    return tok, model

def generate(tok, model, query, context):
    prompt = f"""
You are a scientific fact checker.

You MUST use ONLY the given context.

Return in this format:

LABEL: SUPPORT / REFUTE / NEUTRAL
EVIDENCE: exact sentence from context

If no supporting sentence exists, write:
EVIDENCE: NONE

Context:
{context}

Claim:
{query}
"""
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False
        )

    generated = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(generated, skip_special_tokens=True).strip()

def parse_output(text):
    text = text.upper()

    if "SUPPORT" in text:
        label = "SUPPORT"
    elif "REFUTE" in text or "CONTRADICT" in text:
        label = "REFUTE"
    else:
        label = "NEUTRAL"

    evidence = ""
    if "EVIDENCE:" in text:
        evidence = text.split("EVIDENCE:")[1].strip()

    return label, evidence

# ============================
# 6. TRUE RAG (FIXED)
# ============================


def rag(query, tok, model):
    docs, ids, probs = retrieve(query)

    outputs = []

    for doc, p in zip(docs, probs):
        out = generate(tok, model, query, doc)
        outputs.append((out, p))

    parsed_outputs = [(parse_output(o), p) for o, p in outputs]

    norm_outputs = [(label, p) for (label, _), p in parsed_outputs]
    evidences = [e for (label, e), _ in parsed_outputs]

    # Weighted voting
    score_map = {}
    for label, p in norm_outputs:
        score_map[label] = score_map.get(label, 0) + p

    final = max(score_map, key=score_map.get)

    return docs, ids, probs, [o for o, _ in outputs], final, evidences
    
# ============================
# 7. METRICS
# ============================
def normalize(text):
    text = text.strip().upper()
    if text.startswith("SUPPORT"):
        return "SUPPORT"
    if text.startswith("REFUTE") or text.startswith("CONTRADICT"):
        return "REFUTE"
    return "NEUTRAL"

def compute_em(pred, gt):
    return int(normalize(pred) == normalize(gt))

def compute_span_grounding(docs, evidences):
    grounded = []

    for ev in evidences:
        if ev == "NONE" or len(ev.strip()) == 0:
            grounded.append(0)
            continue

        found = any(ev.lower() in doc.lower() for doc in docs)
        grounded.append(int(found))

    return np.mean(grounded)


# ============================
# 8. hallucination (FIXED)
# ============================
def compute_hallucination_soft(docs, answer, query):
    verdict = normalize(answer)

    # build hypothesis from the MODEL ANSWER
    if verdict == "SUPPORT":
        hypothesis = f"The claim is supported: {query}"
    elif verdict == "REFUTE":
        hypothesis = f"The claim is false: {query}"
    else:
        hypothesis = f"There is insufficient evidence for: {query}"

    inputs = [
        {"text": doc[:512], "text_pair": hypothesis}
        for doc in docs
    ]

    results = nli_model(inputs, truncation=True)

    if verdict == "SUPPORT":
        scores = [r["score"] for r in results if r["label"].upper() == "ENTAILMENT"]

    elif verdict == "REFUTE":
        scores = [r["score"] for r in results if r["label"].upper() == "CONTRADICTION"]

    else:
        return 0.0

    return 1 - max(scores) if scores else 1.0

def compute_hallucination_binary(docs, answer, query):
    verdict = normalize(answer)

    if verdict == "SUPPORT":
        hypothesis = f"The claim is supported: {query}"
    elif verdict == "REFUTE":
        hypothesis = f"The claim is false: {query}"
    else:
        return 0

    inputs = [
        {"text": doc[:512], "text_pair": hypothesis}
        for doc in docs
    ]

    results = nli_model(inputs, truncation=True)

    has_entailment = any(r["label"].upper() == "ENTAILMENT" for r in results)

    return 0 if has_entailment else 1

def compute_final_hallucination(nli_score, span_score):
    return 0.7 * nli_score + 0.3 * (1 - span_score)
    
# ============================
# 9. DATASET PARSER (FIXED)
# ============================
def parse_sample(sample):
    query = sample["claim"]

    label = sample["evidence_label"]

    if label == "SUPPORT":
        gt = "SUPPORT"
    elif label == "CONTRADICT":
        gt = "REFUTE"
    else:
        gt = "NEUTRAL"

    # FIX: handle empty doc_id
    if sample["evidence_doc_id"] == "":
        gold_doc_id = None
    else:
        gold_doc_id = int(sample["evidence_doc_id"])

    return query, gt, gold_doc_id
    
# ============================
# 10. RUN
# ============================
results = []

for dataset_name, config, split, label in DATASETS:
    dataset = load_dataset(dataset_name, config,split=split)

    for model_name, mode in MODELS:
        tok, model = load_model(model_name)

        for i, sample in enumerate(dataset):
            if i >= 5:
                break
            query, gt, gold_doc_id = parse_sample(sample)



            docs, ids, probs, answers, pred, evidences = rag(query, tok, model)

            em = compute_em(pred, gt)
            hallucination_nli = compute_hallucination_soft(docs, pred, query)
            hallucination_bin = compute_hallucination_binary(docs, pred, query)
            span_score = compute_span_grounding(docs, evidences)
            hallucination = compute_final_hallucination(hallucination_nli, span_score)
            # DEBUG BLOCK
            print("QUERY:", query)
            print("GT:", gt, "| PRED:", normalize(pred))
            
            # Debug NLI skipped for quick test
            
            print("hal:", hallucination)
            print("Span Score:", span_score)
            print("NLI Score:", hallucination_nli)
            print("-" * 60)

            if gold_doc_id is None:
                recall = None
            else:
                recall = int(gold_doc_id in ids)
            if recall == 0:
                error_type = "retrieval_failure"
            elif em == 0 and hallucination < 0.5:
                error_type = "reasoning_error"
            elif hallucination > 0.7:
                error_type = "hallucination"
            else:
                error_type = "correct"
            
            results.append({
                "model": model_name,

                "gold_doc_id": gold_doc_id,
                "retrieved_doc_ids": ids,
                "retrieval_recall": recall,
                
                "query": query,
                "ground_truth": gt,
                "prediction": normalize(pred),

                "EM": em,
                "hallucination": hallucination,
                "hallucination_bin": hallucination_bin,
                "error_type": error_type,
                
                "top_doc": docs[0],
                "all_docs": docs,
                "doc_probs": probs.tolist(),
                "answers": answers,
                "span_grounding": span_score,
                "evidences": evidences,
            })

            print(f"{i} | EM:{em} | H:{hallucination:.3f}")

        del model
        del tok
        torch.cuda.empty_cache()
        gc.collect()

# ============================
# 11. SAVE
# ============================
df = pd.DataFrame(results)
df.to_csv("final_rag_results.csv", index=False)

summary = df.groupby("model").agg({
    "EM": "mean",
    "hallucination": "mean",
    "retrieval_recall": "mean"
})
print(summary)
