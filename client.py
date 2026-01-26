import gradio as gr
import requests
import numpy as np
import uuid
import json

# ================= 配置 =================
HTTP_SERVER_URL = "http://127.0.0.1:10095/api/asr_streaming"
AUTH_TOKEN = "Bearer my_secret_token"

class HTTPASRClient:
    def __init__(self):
        self.headers = {"Authorization": AUTH_TOKEN}
        self.session_id = str(uuid.uuid4())
        self.history_text = ""
        self.current_text = ""

    def reset_all(self):
        self.session_id = str(uuid.uuid4())
        self.history_text = ""
        self.current_text = ""
        return ""

    def send_audio(self, audio_data, sr):
        """
        发送音频，并在本地进行降采样以节省带宽
        """
        # === 核心优化：本地降采样 ===
        target_sr = 16000
        
        # 1. 如果是 48k (最常见)，直接切片，数据量减少 2/3，速度起飞
        if sr == 48000:
            step = 3
            audio_data = audio_data[::step]
            actual_fs = 16000
        # 2. 如果是 32k
        elif sr == 32000:
            step = 2
            audio_data = audio_data[::step]
            actual_fs = 16000
        # 3. 如果是 16k，直接发
        elif sr == 16000:
            actual_fs = 16000
        # 4. 其他奇葩采样率 (如 44.1k)，本地切片会导致音质受损或变调
        #    所以直接原样发给 Server，让 Server 用 torchaudio 高质量重采样
        else:
            actual_fs = sr

        # 转 Bytes
        audio_int16 = (audio_data * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        # 发送请求
        files = {'file': ('audio.pcm', audio_bytes, 'application/octet-stream')}
        
        # 告诉 Server 我们实际发过去的是多少采样率
        data = {
            'session_id': self.session_id,
            'is_speaking': 'true',
            'audio_fs': actual_fs 
        }

        try:
            # timeout 设置短一点，保证实时性，如果网络卡了就丢弃这一帧
            response = requests.post(
                HTTP_SERVER_URL, files=files, data=data, headers=self.headers, timeout=2
            )
            if response.status_code == 200:
                res = response.json()
                self.current_text = res.get("text", "")
        except Exception as e:
            # 网络波动是正常的，打印错误但不崩溃
            print(f"Stream Error: {e}")

        return self.history_text + self.current_text

    def finish_sentence(self):
        """停止录音：归档"""
        print("Finishing sentence...")
        try:
            files = {'file': ('empty.pcm', b'', 'application/octet-stream')}
            # 结束信号，audio_fs 填什么都行，不重要
            data = {'session_id': self.session_id, 'is_speaking': 'false', 'audio_fs': 16000}
            
            response = requests.post(
                HTTP_SERVER_URL, files=files, data=data, headers=self.headers, timeout=5
            )
            
            final_sentence = ""
            if response.status_code == 200:
                res = response.json()
                final_sentence = res.get("text", "")
            
            if not final_sentence:
                final_sentence = self.current_text

            self.history_text = ""
            self.current_text = ""
            self.session_id = str(uuid.uuid4()) # 换新 ID
            
        except Exception as e:
            print(f"Finish Error: {e}")
            self.history_text += self.current_text
            self.current_text = ""
            self.session_id = str(uuid.uuid4())
        
        return self.history_text

client = HTTPASRClient()

# ================= Gradio UI =================

def process_audio(audio):
    if audio is None:
        return client.history_text + client.current_text
    
    sr, data = audio
    
    # Gradio 出来的 data 可能是 (samples, channels) 或者是 (samples,)
    # 1. 转单声道
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
        
    # 2. 归一化转 float32 (Gradio 有时给 int16，有时给 float32，统一一下)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    else:
        data = data.astype(np.float32)
        # 如果已经是 float 但范围在 [-32768, 32768] 之间
        if np.abs(data).max() > 1.0:
            data = data / 32768.0
        
    return client.send_audio(data, sr)

def on_stop():
    return client.finish_sentence()

def on_clear():
    return client.reset_all()

with gr.Blocks(title="FunASR 极速版") as demo:
    gr.Markdown("### 智能语音听写 (Bandwidth Optimized)")
    
    with gr.Row():
        with gr.Column(scale=1):
            mic_input = gr.Audio(
                sources=["microphone"], 
                streaming=True, 
                type="numpy",
                label="按住说话"
            )
            clear_btn = gr.Button("🗑️ 清空", variant="stop")
            
        with gr.Column(scale=2):
            output_text = gr.Textbox(
                label="识别结果", 
                lines=20, 
                interactive=False,
                autoscroll=True
            )

    mic_input.stream(process_audio, inputs=[mic_input], outputs=[output_text])
    mic_input.stop_recording(on_stop, outputs=[output_text])
    clear_btn.click(on_clear, outputs=[output_text])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
