# collect_data.py
import os
import wave
import pyaudio
import numpy as np
from datetime import datetime


def record_audio(output_dir, wake_word=None, duration=0.8, sample_rate=16000):
    """录制音频样本用于训练唤醒词模型"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=100)

    print("录音中..." + f"请说{wake_word}" if wake_word else "")
    frames = []

    for _ in range(0, int(sample_rate / 100 * duration)):
        data = stream.read(100)
        frames.append(data)

    print("录音完成")

    stream.stop_stream()
    stream.close()
    p.terminate()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(output_dir, f"sample_{timestamp}.wav")

    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f"保存到 {file_path}")
    return file_path


def collect_wake_word_samples(wake_word, num_samples=100):
    wake_word_dir = f"data/{wake_word}"

    print(f"我们将录制 {num_samples} 个\"{wake_word}\"的样本")
    print("按回车键开始录制每个样本...")

    for i in range(num_samples):
        input(f"按回车键录制样本 {i + 1}/{num_samples}...")
        record_audio(wake_word_dir, wake_word)

    print(f"已收集 {num_samples} 个唤醒词""{wake_word}""的样本")

def collect_negative_samples(num_samples=100):
    negative_dir = "data/negative"

    print(f"现在将录制 {num_samples} 个非唤醒词的样本（随机语音）")
    # print("按回车键开始录制每个样本...")

    for i in range(num_samples):
        # input(f"按回车键录制非唤醒词样本 {i + 1}/{num_samples}...")
        print(f"录制非唤醒词样本 {i + 1}/{num_samples}...")
        record_audio(negative_dir)

    print(f"已收集 {num_samples} 个非唤醒词样本")

if __name__ == "__main__":
    wake_word = input("请输入您的自定义唤醒词: ")

    # 收集唤醒词样本
    num_wake_samples = int(input("要收集多少个唤醒词样本? [100]: ") or "100")
    collect_wake_word_samples(wake_word, num_wake_samples)

    # 收集非唤醒词样本
    num_negative = int(input("要收集多少个非唤醒词样本? [100]: ") or "100")
    collect_negative_samples(num_negative)
