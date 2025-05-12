#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_small.py

より小さなBlipモデルを使用した皮膚疾患診断用学習スクリプト。
ダウンロードサイズが小さく、学習が早く完了します。

使用方法:
python train_small.py --json_path=/path/to/your/data.json
"""

import os
import json
import argparse
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split

from transformers import (
    BlipProcessor, 
    BlipForConditionalGeneration,
    Trainer, 
    TrainingArguments
)

# ========== 多クラス分類対象のクラス一覧 ==========
CLASSES = [
    "ADM",
    "Basal_cell_carcinoma",
    "Ephelis",
    "Malignant_melanoma",
    "Melasma",
    "Nevus",
    "Seborrheic_keratosis",
    "Solar_lentigo"
]
NUM_CLASSES = len(CLASSES)

def get_class_index_from_generated(generated_text: str) -> int:
    """
    生成テキストからクラスを特定する簡易関数。
    CLASSES の疾患名が部分一致すれば、そのクラスを返す (先にマッチした方優先)。
    見つからなければ -1 を返す (未知クラス扱い)。
    """
    gen_lower = generated_text.lower()
    for i, cls_name in enumerate(CLASSES):
        if cls_name.lower() in gen_lower:
            return i
    return -1

def get_class_index_from_label(label_text: str) -> int:
    """
    ラベル文字列からクラスを特定する簡易関数。
    CLASSES の疾患名が部分一致すれば、そのクラスを返す (先にマッチした方優先)。
    見つからなければ -1 を返す。
    """
    label_lower = label_text.lower()
    for i, cls_name in enumerate(CLASSES):
        if cls_name.lower() in label_lower:
            return i
    return -1

class FaceStainMultiModalDataset(Dataset):
    def __init__(self, data_file, processor):
        print(f"JSONデータを読み込んでいます: {data_file}")
        with open(data_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.processor = processor
        self.data = []
        missing_count = 0
        for item in raw_data:
            if (
                isinstance(item, dict)
                and "image" in item
                and "conversations" in item
            ):
                img_path = item["image"]
                if os.path.exists(img_path):
                    self.data.append(item)
                else:
                    missing_count += 1
                    if missing_count <= 5:  # 最初の5つだけ表示
                        print(f"[WARN] 画像ファイルが見つかりません: {img_path}")
                    elif missing_count == 6:
                        print("他にも見つからないファイルがあります...")

        print(f"データセットの読み込み完了: {len(self.data)} サンプル (欠損: {missing_count})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item["image"]
        convs = item["conversations"]
        user_text = convs[0]["value"] if len(convs) > 0 else ""
        assistant_text = convs[1]["value"] if len(convs) > 1 else ""
        
        # 画像を読み込み
        image = Image.open(img_path).convert('RGB')
        
        # プロセッサで処理
        encoding = self.processor(images=image, text=user_text, padding="max_length", truncation=True, return_tensors="pt")
        
        # テンソルから次元を削除
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        # ラベルを追加
        encoding["labels"] = encoding["input_ids"].clone()
        
        # 追加情報
        encoding["text"] = assistant_text  # 後で評価時に使用
        
        return encoding

def main():
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description="小さなBLIPモデルを使用した皮膚疾患診断用学習スクリプト")
    parser.add_argument("--json_path", type=str, required=True,
                        help="学習データのJSONファイルへのパス")
    parser.add_argument("--output_dir", type=str, default="./blip_finetuned",
                        help="学習済みモデルの保存先ディレクトリ")
    parser.add_argument("--epochs", type=int, default=1, 
                        help="学習エポック数")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="バッチサイズ")
    args = parser.parse_args()

    # デバイスの設定
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU (CUDA) を使用します")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Apple Silicon (MPS) を使用します")
    else:
        device = torch.device("cpu")
        print("CPU を使用します")

    # モデルとプロセッサの準備
    model_name = "Salesforce/blip-image-captioning-base"  # より小さなモデル
    print(f"ベースモデルをロードしています: {model_name}")

    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)

    # モデルをGPUに転送
    model.to(device)

    # データセットの準備
    print("\n=== データセットを準備しています ===")
    full_dataset = FaceStainMultiModalDataset(args.json_path, processor)

    # トレーニングデータと評価データに分割
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, eval_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"トレーニングデータ: {len(train_dataset)} サンプル")
    print(f"評価データ: {len(eval_dataset)} サンプル")

    # トレーニング引数の設定
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=5e-5,
        logging_steps=10,
        save_steps=50,
        eval_steps=50,
        evaluation_strategy="steps",
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
    )

    # トレーナーの設定
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # トレーニングの実行
    print("\n=== トレーニングを開始します ===")
    trainer.train()

    # モデルの保存
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"\n学習済みモデルを保存しました: {args.output_dir}")
    print("\nこのモデルを使用するには、アプリのサイドバーでこのパスを指定してください:")
    print(f"絶対パス: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main() 