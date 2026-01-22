# -*- encoding: utf-8 -*-
import os
import time
import json
import uuid
import asyncio
import numpy as np
import torch
import torchaudio.functional as F 
from fastapi import FastAPI, UploadFile, File, Form, Header
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from funasr import AutoModel

# ================= 配置 =================
class Args:
    host = "0.0.0.0"
    port = 10095
    asr_model = "FunAudioLLM/Fun-ASR-Nano-2512"
    vad_model = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
    punc_model = "iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727"
    device = "cuda" 

args = Args()

# ================= 全局状态 =================
session_store = {} 
models = {}
inference_executor = ThreadPoolExecutor(max_workers=10)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("正在加载模型...", flush=True)
    models["asr"] = AutoModel(model=args.asr_model, device=args.device, disable_log=True)
    models["vad"] = AutoModel(model=args.vad_model, device=args.device, disable_log=True)
    models["punc"] = AutoModel(model=args.punc_model, device=args.device, disable_log=True)
    print("模型加载完成！", flush=True)
    yield
    models.clear()

app = FastAPI(lifespan=lifespan)

# ================= 核心逻辑 =================

def decode_and_resample(chunk_bytes, input_fs):
    """解码并强制重采样到 16k"""
    data_int16 = np.frombuffer(chunk_bytes, dtype=np.int16)
    data_float32 = data_int16.astype(np.float32) / 32768.0
    tensor = torch.from_numpy(data_float32)
    
    if input_fs != 16000:
        tensor = tensor.unsqueeze(0) 
        tensor = F.resample(tensor, input_fs, 16000)
        tensor = tensor.squeeze(0)
    return tensor

async def run_inference(model, input_data, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        inference_executor, 
        lambda: model.generate(input=input_data, **kwargs)
    )

@app.post("/api/asr_streaming")
async def asr_streaming(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    is_speaking: str = Form("true"),
    audio_fs: int = Form(16000),
    authorization: str = Header(None)
):
    # 1. 初始化 Session
    if session_id not in session_store:
        session_store[session_id] = {
            "asr_online": {"cache": {}}, # 这里就是 Cache，用于流式连贯性
            "buffer": [],                # 音频缓冲
            "text_segment": "",          # 当前句子的识别结果
        }
    
    session = session_store[session_id]
    
    # 2. 处理音频
    audio_bytes = await file.read()
    if len(audio_bytes) > 0:
        audio_tensor = decode_and_resample(audio_bytes, audio_fs)
        session["buffer"].append(audio_tensor)
        
        # 3. 流式推理 (使用 Cache)
        res_online = await run_inference(
            models["asr"], 
            input=[audio_tensor], 
            **session["asr_online"] # 传入并更新 Cache
        )
        if res_online:
            session["text_segment"] = res_online[0]["text"]

    # 4. 句子结束处理 (Offline 修正)
    is_final = False
    final_result = ""
    
    if is_speaking == "false":
        is_final = True
        if len(session["buffer"]) > 0:
            full_tensor = torch.cat(session["buffer"])
            # 离线高精度推理
            res_offline = await run_inference(models["asr"], input=[full_tensor])
            raw_text = res_offline[0]["text"] if res_offline else ""
            # 加标点
            if raw_text:
                res_punc = await run_inference(models["punc"], input=raw_text)
                final_result = res_punc[0]["text"]
            else:
                final_result = ""
        
        # 清理 Session，准备下一句
        # 注意：我们删除了 session，意味着 Cache 被清空了
        del session_store[session_id]
    else:
        # 流式中间结果
        final_result = session["text_segment"]

    return {
        "text": final_result,
        "is_final": is_final
    }

@app.post("/api/reset")
async def reset_session(session_id: str = Form(...)):
    if session_id in session_store:
        del session_store[session_id]
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
