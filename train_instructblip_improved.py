#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_instructblip_improved.py

InstructBLIPモデルを皮膚疾患診断用にファインチューニングするスクリプト。
元のtrain_instructblip_multiclass_with_metrics_all_curves_improved.pyを簡易化しています。

使用方法:
python train_instructblip_improved.py --json_path=/path/to/your/data.json

JSONデータの形式:
[
  {
    "image": "画像へのパス",
    "conversations": [
      {"value": "ユーザーの質問/指示"},
      {"value": "アシスタントの回答（疾患名を含む）"}
    ]
  },
  ...
]
"""

import os
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split

from transformers import (
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
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

def compute_metrics(eval_preds):
    """
    学習途中のeval時にマクロ指標をモニタするための関数。
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
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # 有効な予測のみにフィルタ
    valid_mask = (y_pred >= 0) & (y_true >= 0)
    if not np.any(valid_mask):
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
        }

    y_pred_filtered = y_pred[valid_mask]
    y_true_filtered = y_true[valid_mask]

    accuracy = accuracy_score(y_true_filtered, y_pred_filtered)
    macro_precision = precision_score(y_true_filtered, y_pred_filtered, average="macro", zero_division=0)
    macro_recall = recall_score(y_true_filtered, y_pred_filtered, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true_filtered, y_pred_filtered, average="macro", zero_division=0)

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }

def main():
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description="InstructBLIPモデルを皮膚疾患診断用にファインチューニングする")
    parser.add_argument("--json_path", type=str, default="/Users/iinuma/Desktop/face_classification_K/annotations/final.json",
                        help="学習データのJSONファイルへのパス")
    parser.add_argument("--output_dir", type=str, default="./instructblip_finetuned_no_image_token",
                        help="学習済みモデルの保存先ディレクトリ")
    parser.add_argument("--epochs", type=int, default=3, 
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

    # データローダーの設定
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: instructblip_collate_fn(b, processor)
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: instructblip_collate_fn(b, processor)
    )

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
        compute_metrics=compute_metrics,
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

    # 混同行列の表示
    print("\n=== 混同行列を作成しています ===")
    predictions_output = trainer.predict(eval_dataset)
    predictions, label_ids = predictions_output.predictions, predictions_output.label_ids

    pred_texts = processor.tokenizer.batch_decode(predictions, skip_special_tokens=True)
    label_texts = processor.tokenizer.batch_decode(
        np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id),
        skip_special_tokens=True
    )

    y_pred, y_true = [], []
    for pt, lt in zip(pred_texts, label_texts):
        pred_idx = get_class_index_from_generated(pt)
        true_idx = get_class_index_from_label(lt)
        y_pred.append(pred_idx)
        y_true.append(true_idx)
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    valid_mask = (y_pred >= 0) & (y_true >= 0)
    y_pred_f = y_pred[valid_mask]
    y_true_f = y_true[valid_mask]

    cm = confusion_matrix(y_true_f, y_pred_f, labels=range(NUM_CLASSES))
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(NUM_CLASSES)
    plt.xticks(tick_marks, CLASSES, rotation=90)
    plt.yticks(tick_marks, CLASSES)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f"{args.output_dir}/confusion_matrix.png")
    plt.close()
    print(f"混同行列を保存しました: {args.output_dir}/confusion_matrix.png")

    # モデルの保存
    trainer.save_model(args.output_dir)
    print(f"\n学習済みモデルを保存しました: {args.output_dir}")
    print("\nこのモデルを使用するには、アプリのサイドバーでこのパスを指定してください:")
    print(f"絶対パス: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main() 