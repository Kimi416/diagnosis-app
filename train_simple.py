#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_simple.py

InstructBLIPモデルを皮膚疾患診断用にファインチューニングする簡易スクリプト。
scikit-learnの依存性を排除し、基本的なモデルトレーニングのみを行います。

使用方法:
python train_simple.py --json_path=/path/to/your/data.json
"""

import os
import json
import argparse
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split

from transformers import (
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
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
    def __init__(self, data_file):
        print(f"JSONデータを読み込んでいます: {data_file}")
        with open(data_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

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
        return (img_path, user_text, assistant_text)

def instructblip_collate_fn(batch, processor):
    images, user_prompts, assistant_responses = [], [], []
    
    for (img_path, user_text, assistant_text) in batch:
        # 画像を読み込みRGBに変換
        img = Image.open(img_path).convert("RGB")
        images.append(img)
        # プロンプトの生成
        prompt = f"USER: {user_text}\nASSISTANT:"
        user_prompts.append(prompt)
        assistant_responses.append(assistant_text)

    # プロセッサで画像とテキストを一括処理
    inputs = processor(
        images=images,
        text=user_prompts,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=512,
    )

    # ラベル（アシスタント応答）のトークナイズ
    labels = processor.tokenizer(
        assistant_responses,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=512,
    )["input_ids"]

    # パディングトークンを -100 にして損失計算から除外
    labels[labels == processor.tokenizer.pad_token_id] = -100
    inputs["labels"] = labels
    
    # Q-Former用の入力（input_idsをコピー）
    inputs["qformer_input_ids"] = inputs["input_ids"].clone()
    inputs["qformer_attention_mask"] = inputs["attention_mask"].clone()

    return inputs

def compute_simple_metrics(eval_preds):
    """
    scikit-learnを使わない簡易的な評価関数
    """
    predictions, label_ids = eval_preds
    global processor

    # decode
    pred_texts = processor.tokenizer.batch_decode(predictions, skip_special_tokens=True)
    label_texts = processor.tokenizer.batch_decode(
        np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id),
        skip_special_tokens=True
    )

    # クラスIDへ変換
    y_pred, y_true = [], []
    for pt, lt in zip(pred_texts, label_texts):
        pred_idx = get_class_index_from_generated(pt)
        true_idx = get_class_index_from_label(lt)
        y_pred.append(pred_idx)
        y_true.append(true_idx)

    # 正解数をカウント
    correct = 0
    valid_count = 0
    
    for p, t in zip(y_pred, y_true):
        if p >= 0 and t >= 0:  # 両方有効な予測である場合
            valid_count += 1
            if p == t:
                correct += 1
    
    accuracy = correct / max(valid_count, 1)  # ゼロ除算防止
    
    return {
        "accuracy": accuracy,
    }

def main():
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description="InstructBLIPモデルを皮膚疾患診断用にファインチューニングする簡易スクリプト")
    parser.add_argument("--json_path", type=str, required=True,
                        help="学習データのJSONファイルへのパス")
    parser.add_argument("--output_dir", type=str, default="./instructblip_finetuned_no_image_token",
                        help="学習済みモデルの保存先ディレクトリ")
    parser.add_argument("--epochs", type=int, default=1, 
                        help="学習エポック数")
    parser.add_argument("--batch_size", type=int, default=1,
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
    model_name = "Salesforce/instructblip-flan-t5-xl"
    print(f"ベースモデルをロードしています: {model_name}")

    global processor
    processor = InstructBlipProcessor.from_pretrained(model_name, use_fast=False)
    model = InstructBlipForConditionalGeneration.from_pretrained(model_name)

    # トークナイザ設定
    if model.config.pad_token_id is None:
        model.config.pad_token_id = processor.tokenizer.pad_token_id
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = (
            processor.tokenizer.bos_token_id 
            if processor.tokenizer.bos_token_id is not None 
            else processor.tokenizer.eos_token_id
        )
    if model.config.eos_token_id is None:
        model.config.eos_token_id = processor.tokenizer.eos_token_id

    # 画像サイズの設定
    processor.image_processor.size = {"height": 224, "width": 224}
    processor.image_processor.crop_size = {"height": 224, "width": 224}

    # モデルをGPUに転送
    model.to(device)
    model.config.use_cache = False

    # データセットの準備
    print("\n=== データセットを準備しています ===")
    full_dataset = FaceStainMultiModalDataset(args.json_path)

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
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=1e-6,
        logging_steps=10,
        save_steps=50,
        fp16=False,
        report_to="none",
        save_total_limit=2,
        eval_strategy="epoch",
        do_eval=True,
        predict_with_generate=True,
    )

    # トレーナーの設定
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=lambda b: instructblip_collate_fn(b, processor),
        compute_metrics=compute_simple_metrics,
    )

    # トレーニングの実行
    print("\n=== トレーニングを開始します ===")
    trainer.train()

    # 評価
    print("\n=== 評価を実行しています ===")
    final_eval_metrics = trainer.evaluate()
    print("\n=== 最終評価結果 ===")
    for key, value in final_eval_metrics.items():
        print(f"{key}: {value:.4f}")

    # モデルの保存
    trainer.save_model(args.output_dir)
    print(f"\n学習済みモデルを保存しました: {args.output_dir}")
    print("\nこのモデルを使用するには、アプリのサイドバーでこのパスを指定してください:")
    print(f"絶対パス: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main() 