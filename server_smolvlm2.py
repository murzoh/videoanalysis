#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SmolVLM2 Remote Server (self-healing, no-Conda)
- Detects GPU + tests a tiny CUDA op.
- If CUDA kernels don't run (e.g., new SM 12.x GPU on old wheels), auto-installs
  newer PyTorch *nightly* CUDA wheels (tries cu130 → cu128 → cu126) via pip,
  then restarts itself.
- Falls back to CPU wheels if CUDA wheels still don't work.
- Disables fragile fused attention paths by default.
- Includes runtime failover to CPU to avoid 500s mid-request.
"""

import os, sys, subprocess, re, io, base64
from typing import List, Optional, Literal

# Make Transformers avoid torchvision internals
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
# Disable fused kernels that may not have SM12.x builds yet
os.environ.setdefault("XFORMERS_DISABLED", "1")
os.environ.setdefault("FLASH_ATTENTION_DISABLE", "1")

# --- Bootstrap: ensure a working PyTorch for this GPU (no conda) ----------------------

def _run(cmd, check=False, capture=True):
    return subprocess.run(cmd, check=check, text=True,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.STDOUT if capture else None)

def _nvidia_smi_summary():
    try:
        out = _run(["nvidia-smi"]).stdout
        m_drv = re.search(r"Driver Version:\s*([\d.]+)", out or "")
        m_cuda = re.search(r"CUDA Version:\s*([\d.]+)", out or "")
        return (m_drv.group(1) if m_drv else "?",
                m_cuda.group(1) if m_cuda else "?",
                out.strip() if out else "")
    except Exception:
        return ("?", "?", "")

def _tiny_cuda_ok_with_current_torch():
    try:
        import torch
        if not torch.cuda.is_available():
            return True, "cuda_unavailable"  # Nothing to fix; we'll run CPU.
        # Print useful info
        dev = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        # Force safest SDPA path
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass
        # Try a tiny matmul on CUDA
        a = torch.randn(256, 256, device="cuda")
        b = a @ a.t()
        torch.cuda.synchronize()
        return True, f"ok({dev}, sm{cap[0]}.{cap[1]})"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def _pip(*pkgs):
    print(f"[bootstrap] pip {' '.join(pkgs)}", file=sys.stderr)
    return _run([sys.executable, "-m", "pip", *pkgs], capture=False)

def _ensure_working_torch():
    # Allow opting out
    if os.environ.get("SMOL_SKIP_AUTOFIX") == "1":
        return
    ok, why = _tiny_cuda_ok_with_current_torch()
    if ok:
        print(f"[bootstrap] CUDA test passed: {why}")
        return
    print(f"[bootstrap] CUDA test failed: {why}", file=sys.stderr)
    drv, cudaver, _ = _nvidia_smi_summary()
    print(f"[bootstrap] nvidia-smi driver={drv}, CUDA={cudaver}", file=sys.stderr)

    # Uninstall any broken wheels
    _pip("uninstall", "-y", "torch", "torchvision", "torchaudio", "xformers", "flash-attn")

    # Try nightlies in descending preference for new GPUs
    nightly_indexes = [
        "https://download.pytorch.org/whl/nightly/cu130",
        "https://download.pytorch.org/whl/nightly/cu128",
        "https://download.pytorch.org/whl/nightly/cu126",
    ]
    for idx in nightly_indexes:
        try:
            _pip("install", "--pre", "--upgrade", "--index-url", idx, "torch", "torchvision")
            ok2, why2 = _tiny_cuda_ok_with_current_torch()
            if ok2:
                print(f"[bootstrap] Success with nightly @ {idx}: {why2}")
                # Mark so we don't loop
                os.environ["SMOL_SKIP_AUTOFIX"] = "1"
                os.execv(sys.executable, [sys.executable] + sys.argv)
            else:
                print(f"[bootstrap] Still failing after install from {idx}: {why2}", file=sys.stderr)
        except Exception as e:
            print(f"[bootstrap] Install from {idx} failed: {e}", file=sys.stderr)

    # Last resort: CPU wheels so the API still runs
    try:
        _pip("install", "--upgrade", "--index-url", "https://download.pytorch.org/whl/cpu",
             "torch", "torchvision")
        print("[bootstrap] Installed CPU-only torch/vision. Service will run on CPU.")
        os.environ["SMOL_FORCE_CPU"] = "1"
        os.environ["SMOL_SKIP_AUTOFIX"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception as e:
        print(f"[bootstrap] CPU install failed: {e}", file=sys.stderr)
        sys.exit(1)

_ensure_working_torch()

# ---- Safe to import torch-dependent libs now -----------------------------------------
import torch
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# ---------------- Model / device selection -------------------------------------------
MODEL_ID = os.environ.get("SMOLVLM2_MODEL", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
API_KEY  = os.environ.get("SMOLVLM_API_KEY")

def pick_device_and_dtype():
    if os.environ.get("SMOL_FORCE_CPU") == "1":
        print("[startup] SMOL_FORCE_CPU=1 → using CPU.")
        return "cpu", torch.float32
    if not torch.cuda.is_available():
        print("[startup] CUDA not available → using CPU.")
        return "cpu", torch.float32
    name = torch.cuda.get_device_name(0)
    cap_major, cap_minor = torch.cuda.get_device_capability(0)
    print(f"[startup] CUDA device: {name} (SM {cap_major}.{cap_minor})")
    if os.environ.get("SMOL_FORCE_FP32") == "1":
        print("[startup] SMOL_FORCE_FP32=1 → using float32 on CUDA.")
        return "cuda", torch.float32
    # Prefer FP32 on unknown/very new arches to avoid half-only kernels
    if cap_major >= 12:
        print("[startup] New SM detected → preferring float32 on CUDA for safety.")
        return "cuda", torch.float32
    if cap_major < 6:
        print("[startup] Older SM detected → using float32 on CUDA.")
        return "cuda", torch.float32
    return "cuda", torch.float16

DEVICE, DTYPE = pick_device_and_dtype()
print(f"Loading {MODEL_ID} on {DEVICE} with dtype={DTYPE}…")

# Keep SDPA safe
try:
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

def load_model(device: str, dtype: torch.dtype):
    m = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
    )
    return m.to(device).eval()

try:
    model = load_model(DEVICE, DTYPE)
except RuntimeError as e:
    msg = str(e).lower()
    if "no kernel image" in msg or "illegal instruction" in msg:
        if DEVICE == "cuda" and DTYPE != torch.float32:
            print("[startup] Retry on CUDA FP32 due to kernel-image error…")
            DEVICE, DTYPE = "cuda", torch.float32
            model = load_model(DEVICE, DTYPE)
        else:
            print("[startup] Falling back to CPU FP32 due to kernel-image error…")
            DEVICE, DTYPE = "cpu", torch.float32
            model = load_model(DEVICE, DTYPE)
    else:
        raise

torch.set_num_threads(max(1, min(8, os.cpu_count() or 4)))
print("Model ready.")

# ---------------- Inference prompt scaffolding ----------------------------------------
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

# ---------------- FastAPI app ---------------------------------------------------------
from fastapi import FastAPI
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
    if req.period:
        parts.append(f"Period: {req.period}.")
    if req.attack_direction_home:
        parts.append(f"Home attacks: {req.attack_direction_home.replace('_','->')}.")
    parts.append("Decide using all frames jointly. Prefer 'save' over 'goal' if keeper contact is visible; use 'none' if cues are weak or conflicting.")
    if req.prompt:
        parts.append(req.prompt.strip())
    return "\n".join(parts)

@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_ID, "device": DEVICE, "dtype": str(DTYPE)}

def _prepare_inputs(pil_images, user_text):
    messages=[
        {"role":"system","content":SERVER_SYSTEM_PROMPT},
        {"role":"user","content":([{"type":"image","pil":im} for im in pil_images] + [{"type":"text","text":user_text}])}
    ]
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=prompt_text, images=pil_images, return_tensors="pt", truncation=True)
    # Move to selected device
    inputs = {k: (v.to(DEVICE) if hasattr(v, "to") else v) for k, v in inputs.items()}
    # Align image tensor dtype with model dtype
    model_dtype = next(model.parameters()).dtype
    if "pixel_values" in inputs and hasattr(inputs["pixel_values"], "dtype") and inputs["pixel_values"].dtype != model_dtype:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model_dtype)
    return inputs

def _run_generate(inputs, max_new_tokens: int):
    with torch.no_grad():
        return model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max(64, min(256, max_new_tokens))
        )

@app.post("/analyze")
def analyze(req: AnalyzeRequest, x_api_key: Optional[str] = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    try:
        pil_images=[Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB") for b64 in req.images_b64]
        user_text=build_user_text(req)
        inputs=_prepare_inputs(pil_images, user_text)
        out_ids=_run_generate(inputs, req.max_new_tokens)
        text=processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
        return {"text": text}
    except RuntimeError as e:
        msg = str(e).lower()
        if ("no kernel image" in msg or "illegal instruction" in msg) and DEVICE == "cuda":
            # Runtime failover
            print("[runtime] CUDA kernel-image error → switching to CPU FP32 and retrying once…", file=sys.stderr)
            global DEVICE, DTYPE, model
            DEVICE, DTYPE = "cpu", torch.float32
            model = load_model(DEVICE, DTYPE)
            try:
                pil_images=[Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB") for b64 in req.images_b64]
                user_text=build_user_text(req)
                inputs=_prepare_inputs(pil_images, user_text)
                out_ids=_run_generate(inputs, req.max_new_tokens)
                text=processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
                return {"text": text, "fallback": "cpu_fp32"}
            except Exception as e2:
                raise HTTPException(status_code=500, detail=f"Server error after CPU fallback: {type(e2).__name__}: {e2}")
        raise HTTPException(status_code=500, detail=f"Server error: {type(e).__name__}: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server_smolvlm2:app", host="0.0.0.0", port=8000)
