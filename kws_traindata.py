# train_model.py
import os
import pickle

from torchaudio.transforms import Spectrogram
from torchaudio.transforms import MelSpectrogram
import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten, Input
from tensorflow.keras.utils import to_categorical
from typing import Iterable
import librosa
import matplotlib.pyplot as plt
from kws_model import kwsmodel
import tensorflowjs as tfjs
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import getopt
import sys

# 设置参数
SAMPLE_RATE = 16000
DURATION = 0.2  # 秒
HOP_LENGTH = 256
MFCC_FEATURES = 40
N_FFT = 512
DATA_DIR = "data"

device = "cuda:0"
spectrogram = Spectrogram(n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=N_FFT).to(device)
mel_spectrogram = MelSpectrogram(n_mels=MFCC_FEATURES, sample_rate=SAMPLE_RATE,
                                n_fft=N_FFT, norm="slaney", f_min=0, f_max=SAMPLE_RATE / 2,
                                hop_length=HOP_LENGTH, win_length=N_FFT)
mel_spectrogram.to(device)

# Calculate Zero Mean Unit Variance
class ZmuvTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("total", torch.zeros(1))
        self.register_buffer("mean", torch.zeros(1))
        self.register_buffer("mean2", torch.zeros(1))

    def update(self, data, mask=None):
        with torch.no_grad():
            if mask is not None:
                data = data * mask
                mask_size = mask.sum().item()
            else:
                mask_size = data.numel()
            self.mean = (data.sum() + self.mean * self.total) / (self.total + mask_size)
            self.mean2 = ((data ** 2).sum() + self.mean2 * self.total) / (self.total + mask_size)
            self.total += mask_size

    def initialize(self, iterable: Iterable[torch.Tensor]):
        for ex in iterable:
            self.update(ex)

    @property
    def std(self):
        return (self.mean2 - self.mean ** 2).sqrt()

    def forward(self, x):
        return (x - self.mean) / self.std

zmuv_transform = ZmuvTransform().to(device)

def plot_spectrogram(filename, spec, title=None, ylabel="freq_bin", aspect="auto", xmax=None):
    mel_fig, mel_axs = plt.subplots(1, 1)
    mel_axs.set_title(title or "Spectrogram (db)")
    mel_axs.set_ylabel(ylabel)
    mel_axs.set_xlabel("frame")
    im = mel_axs.imshow(librosa.power_to_db(spec), origin="lower", aspect=aspect)
    if xmax:
        mel_axs.set_xlim((0, xmax))

    mel_fig.colorbar(im, ax=mel_axs)

    plt.savefig(filename, format="jpg")
    # my_stringIObytes = io.BytesIO()
    # plt.savefig(my_stringIObytes, format="jpg")
    # my_stringIObytes.seek(0)
    # my_base64_jpgData = base64.b64encode(my_stringIObytes.read())
    # result["plot"] = my_base64_jpgData.decode("utf-8")

def extract_features(file_path, with_plot=False, labels=None, label_infos=None):
    """从音频文件中提取MFCC特征"""
    try:
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
        results = []
        if label_infos is None:
            label_infos = [{
                'start': n * DURATION,
                'end': (n+1) * DURATION,
                'labels': [
                    None
                ]
            } for n in range(min(3, int(len(audio) / SAMPLE_RATE / DURATION)))]
        for label in label_infos:
            if 'start' in label:
                if label['end'] - label['start'] - 0.0001 > DURATION:
                    print(file_path, "\033[31m duration %.2f \033[0m" % (label['end'] - label['start']))
                sample_audio = audio[int(label['start'] * SAMPLE_RATE):int(label['end'] * SAMPLE_RATE)]
            else:
                sample_audio = audio
            # 确保长度一致（填充或截断）
            if len(sample_audio) < SAMPLE_RATE * DURATION:
                sample_audio = np.pad(sample_audio, (0, int(SAMPLE_RATE * DURATION) - len(sample_audio)))
            else:
                sample_audio = sample_audio[:int(SAMPLE_RATE * DURATION)]
            inp = torch.from_numpy(sample_audio).float().to(device)
            if with_plot:
                hey_spectrogram = spectrogram(inp.float())
                plot_spectrogram(file_path[:-4] + ".png", hey_spectrogram.cpu())
            log_mels = mel_spectrogram(inp.float()).add_(1e-4).log_().contiguous()
            for label_ in label['labels']:
                results.append((log_mels.cpu().numpy(), labels.index(label_) if label_ is not None and labels is not None else -1))

        return results
    except Exception as e:
        print(f"处理文件 {file_path} 时出错 {e.__class__.__name__}: {str(e)}")
        return None


def load_data(wake_word):
    """加载并准备训练数据"""

    wake_word_dir = os.path.join(DATA_DIR, wake_word)
    negative_dir = os.path.join(DATA_DIR, "negative")

    features = []
    labels = []

    if os.path.exists(os.path.join(DATA_DIR, wake_word+".pickle")):
        with open(os.path.join(DATA_DIR, wake_word+".pickle"), "rb") as f:
            features, labels = pickle.load(f)
    else:
        label_infos = {}
        for file_name in os.listdir(wake_word_dir):
            if not file_name.endswith('.json'):
                continue
            label_info = json.load(open(os.path.join(wake_word_dir, file_name), "r"))
            for label in label_info:
                file_name = os.path.basename(label['audio'])
                if 'label' in label:
                    label_infos[file_name] = label['label']

        # 处理唤醒词样本（标签1）
        for file_name in tqdm(os.listdir(wake_word_dir), desc="Loading active word"):
            if not file_name.endswith('.wav'):
                continue
            file_path = os.path.join(wake_word_dir, file_name)
            mfccs = extract_features(file_path, True, wake_word, label_infos.get(file_name, None))
            if mfccs is not None:
                for (mfcc, label) in mfccs:
                    features.append(mfcc)
                    labels.append(label)  # 唤醒词标签

        with open(os.path.join(DATA_DIR, wake_word + ".pickle"), "wb") as f:
            pickle.dump((features, labels), f)

    negative_features = []
    negative_labels = []

    if os.path.exists(os.path.join(DATA_DIR, "negative.pickle")):
        with open(os.path.join(DATA_DIR, "negative.pickle"), "rb") as f:
            negative_features, negative_labels = pickle.load(f)
    else:
        # 处理非唤醒词样本（标签0）
        for file_name in tqdm(os.listdir(negative_dir), desc="Loading negative word"):
            if not file_name.endswith('.wav'):
                continue
            file_path = os.path.join(negative_dir, file_name)
            mfccs = extract_features(file_path, labels=None)
            if mfccs is not None:
                for mfcc, label in mfccs:
                    negative_features.append(mfcc)
                    negative_labels.append(len(wake_word))  # 非唤醒词标签

        with open(os.path.join(DATA_DIR, "negative.pickle"), "wb") as f:
            pickle.dump((negative_features, negative_labels), f)

    features.extend(negative_features)
    labels.extend(negative_labels)

    # 转换为numpy数组
    X = np.array(features)
    y = np.array(labels)

    # 重塑数据以适应CNN: (样本数, 高度, 通道)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

    # 转换标签为分类格式
    y = to_categorical(y, num_classes=len(wake_word)+1)

    return X, y


def build_model(input_shape, catanum=2):
    """构建用于唤醒词检测的CNN模型，明确指定输入形状"""
    model = kwsmodel(input_shape=input_shape, catanum=catanum)
    model.compile(loss="binary_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])

    return model

def train_wake_word_model(wake_word, epochs=5):
    """训练并保存唤醒词检测模型"""
    # 加载并准备数据
    X, y = load_data(wake_word)

    # X = np.transpose(np.mean(X.T, axis=1))

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, random_state=42)

    # 设置早停和学习率降低的回调
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
    ]

    # 构建模型
    model = build_model(input_shape=(X.shape[1], X.shape[2], 1), catanum=len(wake_word)+1)

    # 显示模型摘要以验证输入形状
    model.summary()

    # 训练模型
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=32,
        callbacks=callbacks,
        epochs=epochs,
        verbose=1
    )

    # 绘制训练历史
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.title('Accuracy')
    plt.xlabel('Interation')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.title('Loss')
    plt.xlabel('Interation')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')

    # 保存模型
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 保存为TensorFlow SavedModel格式
    model.save(os.path.join(model_dir, f"{wake_word}_model.h5"))
    tfjs.converters.save_keras_model(model, "assets/kws_js/")

    # 测试模型
    loss, accuracy = model.evaluate(X_val, y_val, verbose=1)
    print(f"验证准确率: {accuracy * 100:.2f}%")

    for i in range(y.shape[-1]-1):
        true_pred = model.predict(X[y[:, i] == 1])[:, i]
        print("True %d: %.4f/%.4f/%.4f std: %.4f" % (i, np.min(true_pred), np.mean(true_pred), np.max(true_pred), np.std(true_pred)))

    false_pred = model.predict(X[y[:, y.shape[-1]-1] == 0])[:, y.shape[-1]-1]
    print("False: %.4f/%.4f/%.4f std: %.4f" % (np.min(false_pred), np.mean(false_pred), np.max(false_pred), np.std(false_pred)))

    return X, y, model


if __name__ == "__main__":
    wake_word = None
    epochs = 5
    opts, args = getopt.gnu_getopt(sys.argv[1:], 'w:e:', ['wake_word=', 'epochs='])

    for k, v in opts:
        if k in ("-w", "--wake_word="):
            wake_word = v
        if k in ("-e", "--epochs="):
            epochs = int(v)

    if wake_word is None:
        print("No input wake word")
        sys.exit()

    X, y, model = train_wake_word_model(wake_word, epochs=epochs)
