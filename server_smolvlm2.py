#!/usr/bin/env python3
# --- Permanent fix: enforce known-good Torch/TorchVision and auto-heal if needed ---
import os, sys, subprocess, re

# Keep Transformers from importing torchvision internals (avoids nms path)
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

REQUIRED_TORCH_BASE = "2.5.1"
REQUIRED_VISION_BASE = "0.20.1"
WHEEL_INDEX = os.environ.get("SMOL_TORCH_INDEX", "https://download.pytorch.org/whl/cu121")
PIP_FLAGS = os.environ.get("SMOL_PIP_FLAGS", "--break-system-packages --timeout 300 --retries 5")

def _base_ver(v: str) -> str:
    v = v.split("+", 1)[0]
    v = re.split(r"[- ]", v, 1)[0]
    return v

def _ensure_known_good_pair():
    try:
        import torch
        torch_base = _base_ver(getattr(torch, "__version__", ""))
        try:
            import torchvision
            vision_base = _base_ver(getattr(torchvision, "__version__", ""))
        except Exception as e:
            vision_base = None
            raise RuntimeError(f"torchvision issue: {e}")
        if torch_base == REQUIRED_TORCH_BASE and vision_base == REQUIRED_VISION_BASE:
            return
        raise RuntimeError(f"bad pair torch={torch_base!r} torchvision={vision_base!r}")
    except Exception as e:
        if os.environ.get("SMOL_AUTOFIX_DONE") == "1":
            print(f"[startup] Torch pair still wrong after autofix: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"[startup] Fixing Torch pair due to: {e}", file=sys.stderr)
        subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"])
        cmd = [sys.executable, "-m", "pip", "install", *PIP_FLAGS.split(),
               "--index-url", WHEEL_INDEX,
               f"torch=={REQUIRED_TORCH_BASE}", f"torchvision=={REQUIRED_VISION_BASE}"]
        subprocess.check_call(cmd)
        os.environ["SMOL_AUTOFIX_DONE"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)

_ensure_known_good_pair()

# ---- Safe to import torch-dependent libs now ----
import io, base64
from typing import List, Optional, Literal
import torch
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# ---------------- Model / device ----------------
MODEL_ID = os.environ.get("SMOLVLM2_MODEL", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE    = torch.float16 if DEVICE == "cuda" else torch.float32
API_KEY  = os.environ.get("SMOLVLM_API_KEY")


print(f"Loading {MODEL_ID} on {DEVICE}…")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, torch_dtype=DTYPE, low_cpu_mem_usage=True, trust_remote_code=True
).to(DEVICE)
model.eval()
torch.set_num_threads(max(1, min(8, os.cpu_count() or 4)))
print("Model ready.")

EVENT_ENUM = ["goal","shot_on_target","shot_off_target","save","foul","offside","corner","none"]

SERVER_SYSTEM_PROMPT = f"""
You analyze sequences of football (soccer) frames.

RETURN FORMAT
- Return ONLY a JSON ARRAY (possibly empty). No prose, no code fences.
- For each item you MUST CHOOSE exactly one value for each field.

SCHEMA (choose ONE value per field):
{{
  "event_type": one of {EVENT_ENUM},
  "team": "home" | "away" | "unknown",
  "start_time": number (seconds within the provided window),
  "end_time": number (seconds within the provided window, >= start_time),
  "confidence": number (0.0..1.0),
  "notes": short generic string (no player names/numbers or minute stamps)
}}

STRICT RULES
- If evidence is insufficient, output [] or an item with "event_type":"none" and "team":"unknown".
- DO NOT invent player names, shirt numbers, minutes, or scorelines.
- Timestamps MUST lie inside the queried window and reflect visible action.
- Treat both teams equally; prefer team=unknown if kits unclear.
- 'goal' only if the ball clearly crosses the line; 'save' when keeper stops a goal-bound shot.
""".strip()

app = FastAPI(title="SmolVLM2 Remote Server")

class AnalyzeRequest(BaseModel):
    prompt: Optional[str] = None
    images_b64: List[str]
    max_new_tokens: int = 200
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    home_kit: Optional[str] = None
    away_kit: Optional[str] = None
    period: Optional[Literal["1H","2H","ET1","ET2"]] = None
    attack_direction_home: Optional[Literal["left_to_right","right_to_left"]] = None

def build_user_text(req: AnalyzeRequest) -> str:
    parts=[]
    if req.start_time is not None and req.end_time is not None:
        parts.append(f"Analyze frames from {req.start_time:.2f}s to {req.end_time:.2f}s (timestamps must be inside this window).")
    if req.home_team or req.away_team:
        parts.append(f"Teams: home={req.home_team or 'home'}, away={req.away_team or 'away'}.")
    if req.home_kit or req.away_kit:
        parts.append(f"Kits: home={req.home_kit or 'unknown'}, away={req.away_kit or 'unknown'}.")
    if req.period: parts.append(f"Period: {req.period}.")
    if req.attack_direction_home: parts.append(f"Home attacks: {req.attack_direction_home.replace('_','->')}.")
    parts.append("Decide using all frames jointly. Prefer 'save' over 'goal' if keeper contact is visible; use 'none' if cues are weak or conflicting.")
    if req.prompt: parts.append(req.prompt.strip())
    return "\n".join(parts)

@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_ID, "device": DEVICE}

@app.post("/analyze")
def analyze(req: AnalyzeRequest, x_api_key: Optional[str] = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    try:
        pil_images=[Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB") for b64 in req.images_b64]
        user_text=build_user_text(req)
        messages=[{"role":"system","content":SERVER_SYSTEM_PROMPT},
                  {"role":"user","content":([{"type":"image","pil":im} for im in pil_images]+[{"type":"text","text":user_text}])}]
        prompt_text=processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs=processor(text=prompt_text, images=pil_images, return_tensors="pt", truncation=True).to(DEVICE)
        if "pixel_values" in inputs and inputs["pixel_values"].dtype!=model.dtype:
            inputs["pixel_values"]=inputs["pixel_values"].to(dtype=model.dtype)
        with torch.no_grad():
            out_ids=model.generate(**inputs, do_sample=False, max_new_tokens=max(64, min(256, req.max_new_tokens)))
        text=processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server_smolvlm2:app", host="0.0.0.0", port=8000)
