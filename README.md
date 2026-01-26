# funasr-nano-streaming
streaming codes for funasr-nano

欢迎pr

参考: https://github.com/fengin/Fun-ASR-Nano-2512-Deploy

用gemini3-pro改成了http，加了gradio流式输入显示功能
#不是最终版，llm需要一些前一个chunk的文本，参考这里的训练数据准备: https://github.com/modelscope/FunASR/blob/36656aa8f7f30cece85ceda7e2c6ec5952a59925/funasr/datasets/fun_asr_datasets/datasets.py#L237

流式模型代码更新了，新增了prev_text参数：https://github.com/FunAudioLLM/Fun-ASR/commit/c38d22f5f2d82b545ea4dfa8807d6d4c1c826e99

chunk_size可以按需求自己调整下

generate函数调用的时候是可以配置llm推理超参的，记得调一下，还有itn、语种这些超参
