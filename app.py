# app.py
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

PORT = int(os.environ.get("PORT", 8080))

ART_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
FAISS_PATH = os.path.join(ART_DIR, "faiss.index")
META_PATH = os.path.join(ART_DIR, "nco_meta.parquet")
MODEL_NAME_PATH = os.path.join(ART_DIR, "model_name.txt")

app = FastAPI(title="OccuMatch AI - NCO Semantic Search API", version="0.2.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals filled lazily to avoid startup timeout
model: Optional[object] = None
index: Optional[object] = None
meta: Optional[object] = None

def load_artifacts_if_needed():
    global model, index, meta
    if model is not None and index is not None and meta is not None:
        return

    # Validate files exist
    for p in (FAISS_PATH, META_PATH, MODEL_NAME_PATH):
        if not os.path.exists(p):
            raise RuntimeError(f"Missing required artifact: {p}")

    # Import heavy deps only when needed
    import faiss
    import pandas as pd
    from sentence_transformers import SentenceTransformer

    meta = pd.read_parquet(META_PATH)
    index = faiss.read_index(FAISS_PATH)
    with open(MODEL_NAME_PATH, "r") as f:
        model_name = f.read().strip()
    model = SentenceTransformer(model_name)

class SearchRequest(BaseModel):
    query: str
    k: int = 5
    min_confidence: float = 0.0  # percent 0-100

@app.post("/search")
def search(req: SearchRequest):
    load_artifacts_if_needed()

    q = (req.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty query")

    import numpy as np  # local import to keep import graph light at boot

    q_emb = model.encode([q], normalize_embeddings=True).astype("float32")
    scores, idx = index.search(q_emb, req.k)
    raw_scores = scores.flatten()
    idx = idx.flatten()

    results = []
    for i, ridx in enumerate(idx.tolist()):
        if ridx < 0 or ridx >= len(meta):
            continue
        rec = meta.iloc[ridx]
        conf_pct = float(raw_scores[i] * 100.0)
        if conf_pct < req.min_confidence:
            continue
        results.append(
            {
                "code_2015": rec["NCO-2015"],
                "code_2004": rec["NCO-2004"],
                "title": rec["Title"],
                "description": rec["Description"],
                "confidence": conf_pct,
            }
        )

    return {"query": q, "count": len(results), "results": results}

@app.get("/", response_class=HTMLResponse)
def root():
    return """
<!doctype html>
<html><head><meta charset="utf-8"><title>NCO Search</title>
<style>
body{font-family:Arial,sans-serif;padding:16px;max-width:1100px;margin:auto}
input,button{padding:8px;font-size:14px}
table{border-collapse:collapse;margin-top:12px;width:100%}
th,td{border:1px solid #ddd;padding:8px;vertical-align:top}
th{background:#f5f5f5}
.small{color:#666;font-size:12px}
.flex{display:flex;gap:8px;align-items:center;margin:8px 0;flex-wrap:wrap}
</style></head><body>
<h3>OccuMatch AI - NCO Semantic Search</h3>
<div class="small">Search in English or Hindi. Codes are fixed-width: NCO-2015 XXXX.XXXX, NCO-2004 XXXX.XX</div>
<div class="flex">
  <input id="q" placeholder="e.g., tailor, cow herder, गाय पालने वाला" size="50"/>
  <label>Top K</label><input id="k" type="number" value="5" min="1" max="50" style="width:70px"/>
  <label>Min %</label><input id="minc" type="number" value="0" min="0" max="100" style="width:80px"/>
  <button id="go">Search</button>
</div>
<div id="status" class="small"></div>
<table id="results" style="display:none">
  <thead><tr>
    <th>NCO-2015</th><th>NCO-2004</th><th>Title</th><th>Description</th><th>Confidence</th>
  </tr></thead>
  <tbody></tbody>
</table>
<script>
const q=document.getElementById('q'),k=document.getElementById('k'),m=document.getElementById('minc'),
s=document.getElementById('status'),tbl=document.getElementById('results'),tb=tbl.querySelector('tbody');
async function search(){
  const query=q.value.trim(); if(!query){s.textContent='Enter a query.';return;}
  s.textContent='Searching...'; tb.innerHTML=''; tbl.style.display='none';
  try{
    const resp=await fetch('/search',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({query,k:parseInt(k.value),min_confidence:parseFloat(m.value)})});
    if(!resp.ok){s.textContent='Server error '+resp.status;return;}
    const data=await resp.json();
    if(!data.results.length){s.textContent='No results.';return;}
    for(const r of data.results){
      const tr=document.createElement('tr');
      tr.innerHTML=`<td>${r.code_2015}</td><td>${r.code_2004}</td>
        <td>${r.title}</td><td>${r.description}</td>
        <td>${r.confidence.toFixed(2)}%</td>`;
      tb.appendChild(tr);
    }
    s.textContent=''; tbl.style.display='';
  } catch(e){ s.textContent='Error: '+e.message; }
}
document.getElementById('go').addEventListener('click',search);
q.addEventListener('keydown',e=>{if(e.key==='Enter')search();});
</script>
</body></html>
"""

@app.get("/health")
def health():
    try:
        cnt = index.ntotal if index is not None else 0
        rows = len(meta) if meta is not None else 0
        return {"status": "ok", "index_vectors": cnt, "meta_rows": rows}
    except Exception as e:
        return {"status":"error","detail":str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=PORT)
