# ============================
# 1. IMPORTS
# ============================
import os, gc
import numpy as np
import pandas as pd
import torch, faiss

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import evaluate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================
# 2. CONFIG
# ============================
DATASET_NAME = "nvidia/TechQA-RAG-Eval"
TOP_K = 3

MODELS = [
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
]

# ============================
# 3. LOAD DATA
# ============================
full_data = load_dataset(DATASET_NAME, split="train", trust_remote_code=True)
dataset   = full_data.select(range(50))   # eval slice

# ============================
# 4. BUILD CORPUS  ← FIXED
# ============================
docs_db = []

for row in full_data:
    for ctx in row.get("contexts", []):          # contexts is a list of dicts
        txt = ctx.get("text", "")
        if isinstance(txt, str) and len(txt.strip()) > 20:
            docs_db.append(txt[:1500])

# De-duplicate
docs_db = list(dict.fromkeys(docs_db))
print("Total valid docs:", len(docs_db))

# ============================
# 5. EMBEDDINGS + FAISS
# ============================
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

emb = embed_model.encode(
    docs_db,
    batch_size=128,
    convert_to_numpy=True,
    show_progress_bar=True,
).astype("float32")

print("Embedding shape:", emb.shape)
assert len(emb.shape) == 2, f"Bad embedding shape: {emb.shape}"

faiss.normalize_L2(emb)
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)
print("FAISS index built:", index.ntotal, "vectors")

# ============================
# 6. RETRIEVE
# ============================
def retrieve(query):
    q = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q)
    scores, idx = index.search(q, TOP_K)
    docs = [docs_db[i] for i in idx[0]]
    return docs, scores[0]

# ============================
# 7. GENERATOR
# ============================
def load_model(name):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    return tok, model


def generate(tok, model, query, context):
    prompt = (
        "You are a helpful technical assistant.\n"
        "Answer the question using ONLY the context below.\n"
        "If the answer is not in the context, say: I don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )
    inp = tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )

    generated = out[0][inp["input_ids"].shape[1]:]
    return tok.decode(generated, skip_special_tokens=True).strip()

# ============================
# 8. METRICS
# ============================
def normalize(text):
    return text.lower().strip()

def compute_em(pred, gt):
    return int(normalize(gt) in normalize(pred))

def compute_f1(pred, gt):
    pred_tokens = set(normalize(pred).split())
    gt_tokens   = set(normalize(gt).split())
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = pred_tokens & gt_tokens
    if not common:
        return 0.0
    p = len(common) / len(pred_tokens)
    r = len(common) / len(gt_tokens)
    return 2 * p * r / (p + r)

def compute_recall_embedding(retrieved_docs, gt_contexts):
    if not gt_contexts:
        return None
    doc_emb = embed_model.encode(retrieved_docs, convert_to_numpy=True).astype("float32")
    ctx_emb = embed_model.encode(gt_contexts,    convert_to_numpy=True).astype("float32")
    sims = np.matmul(doc_emb, ctx_emb.T)
    return int(np.max(sims) > 0.6)

# ============================
# NEW METRICS
# ============================
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")
smooth = SmoothingFunction().method1

def compute_bleu(pred, gt):
    return sentence_bleu([gt.split()], pred.split(), smoothing_function=smooth)

def compute_rouge(pred, gt):
    scores = rouge.compute(predictions=[pred], references=[gt])
    return scores["rouge1"], scores["rougeL"]

def compute_bertscore(pred, gt):
    res = bertscore.compute(predictions=[pred], references=[gt], lang="en")
    return res["precision"][0], res["recall"][0], res["f1"][0]

# ============================
# 9. NLI  (Hallucination)
# ============================
nli_model = pipeline(
    "text-classification",
    model="facebook/bart-large-mnli",
    device=0,
    torch_dtype=torch.float16,
)

def compute_hallucination(context, answer):
    result = nli_model([{"text": context[:512], "text_pair": answer}])[0]
    return 1 - result["score"] if result["label"].upper() == "ENTAILMENT" else 1.0

# ============================
# 10. MAIN LOOP
# ============================
results = []

for model_name in MODELS:
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print('='*60)

    tok, model = load_model(model_name)

    for i, sample in enumerate(dataset):
        query       = sample.get("question", "")
        gt_answer   = sample.get("answer",   "")
        # FIX: contexts is a list of dicts with a "text" key
        gt_contexts = [c["text"] for c in sample.get("contexts", []) if c.get("text")]

        if not query.strip():
            continue

        # Skip unanswerable questions
        if sample.get("is_impossible", False):
            continue

        docs, scores = retrieve(query)
        context = "\n\n".join(docs)
        pred = generate(tok, model, query, context)

        em  = compute_em(pred, gt_answer)
        f1  = compute_f1(pred, gt_answer)
        recall = compute_recall_embedding(docs, gt_contexts)
        hallucination = compute_hallucination(context, pred)

        bleu = compute_bleu(pred, gt_answer)
        rouge1, rougeL = compute_rouge(pred, gt_answer)
        bert_p, bert_r, bert_f1 = compute_bertscore(pred, gt_answer)

        print(f"[{i}] EM:{em} | F1:{f1:.2f} | BERT:{bert_f1:.2f} | Hal:{hallucination:.3f} | Q: {query[:60]}")

        results.append({
            "model":         model_name,
            "query":         query,
            "prediction":    pred,
            "answer":        gt_answer,
            "EM":            em,
            "F1":            f1,
            "Recall@K":      recall,
            "Hallucination": hallucination,
            "Confidence":    0.5 * float(np.max(scores)) + 0.5 * (1 - hallucination),
            "BLEU": bleu,
            "ROUGE-1": rouge1,
            "ROUGE-L": rougeL,
            "BERT-P": bert_p,
            "BERT-R": bert_r,
            "BERT-F1": bert_f1,
        })

    del model, tok
    torch.cuda.empty_cache()
    gc.collect()

# ============================
# 11. SAVE & SUMMARISE
# ============================
df = pd.DataFrame(results)
df.to_csv("rag_results_techqa.csv", index=False)

summary = df.groupby("model").agg({
    "EM": "mean",
    "F1": "mean",
    "Recall@K": "mean",
    "Hallucination": "mean",
    "Confidence": "mean",
    "BLEU": "mean",
    "ROUGE-1": "mean",
    "ROUGE-L": "mean",
    "BERT-F1": "mean",
})

print("\nSUMMARY:\n", summary)
summary.to_csv("summary_techqa.csv")