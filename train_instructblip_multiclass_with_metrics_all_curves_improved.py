#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_instructblip_multiclass_with_metrics_all_curves_improved.py

改良ポイント:
- 学習終了後に predict() で推論を行い、以下を行う:
  1) ROC曲線 (すべてのクラスを1つの図に)
  2) PR曲線 (すべてのクラスを1つの図に)
  3) 混同行列の可視化 (ヒートマップ)
  4) 疾患別の 精度(Precision)/感度(Recall)/特異度(Specificity)/F1 を一覧表で出力 & CSV保存
  5) さらにマクロ平均(AUC含む) も出力
- 途中の compute_metrics は最終的なマクロ指標をモニタする用途に留め、
  詳細分析は学習後の predict() の結果を使って行う。
"""

import os
import json
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
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc
)
from sklearn.preprocessing import label_binarize

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
        with open(data_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.data = []
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
                    print(f"[WARN] Missing file => skip: {img_path}")

        print(f"Loaded {len(self.data)} samples from {data_file}.")

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
    最終的な詳細分析は学習後の predict() で実施するので、ここはマクロ指標だけ返す。
    """
    predictions, label_ids = eval_preds

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
            "macro_sensitivity": 0.0,
            "macro_specificity": 0.0,
            "macro_roc_auc": 0.0,
            "macro_pr_auc": 0.0,
        }

    y_pred_filtered = y_pred[valid_mask]
    y_true_filtered = y_true[valid_mask]

    accuracy = accuracy_score(y_true_filtered, y_pred_filtered)
    macro_precision = precision_score(y_true_filtered, y_pred_filtered, average="macro", zero_division=0)
    macro_recall = recall_score(y_true_filtered, y_pred_filtered, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true_filtered, y_pred_filtered, average="macro", zero_division=0)

    cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=range(NUM_CLASSES))
    sensitivities, specificities = [], []
    for i in range(NUM_CLASSES):
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        TN = np.sum(cm) - (TP + FP + FN)
        if TP + FN > 0:
            sensitivities.append(TP / (TP + FN))
        if TN + FP > 0:
            specificities.append(TN / (TN + FP))
    macro_sensitivity = np.mean(sensitivities) if sensitivities else 0.0
    macro_specificity = np.mean(specificities) if specificities else 0.0

    y_true_bin = label_binarize(y_true_filtered, classes=range(NUM_CLASSES))
    y_pred_bin = label_binarize(y_pred_filtered, classes=range(NUM_CLASSES))
    try:
        macro_roc_auc = roc_auc_score(y_true_bin, y_pred_bin, average="macro", multi_class="ovr")
    except ValueError:
        macro_roc_auc = 0.0

    pr_aucs = []
    for i in range(NUM_CLASSES):
        precision_i, recall_i, _ = precision_recall_curve(y_true_bin[:, i], y_pred_bin[:, i])
        if len(precision_i) > 1 and len(recall_i) > 1:
            pr_aucs.append(auc(recall_i, precision_i))
    macro_pr_auc = np.mean(pr_aucs) if pr_aucs else 0.0

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "macro_sensitivity": macro_sensitivity,
        "macro_specificity": macro_specificity,
        "macro_roc_auc": macro_roc_auc,
        "macro_pr_auc": macro_pr_auc,
    }

def analyze_final_results(trainer, dataset, output_prefix="final"):
    """
    学習完了後に trainer.predict() を使って詳細分析を行い、以下を出力/保存:
      1) 全クラスまとめてのROC曲線 (1つの図)
      2) 全クラスまとめてのPR曲線 (1つの図)
      3) Confusion Matrix のヒートマップ
      4) 疾患別の指標(Precision, Recall, Specificity, F1) 一覧表 (CSVにも保存)
      5) マクロ平均のスコアも最後に表示
    """
    predictions_output = trainer.predict(dataset)
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

    if len(y_pred_f) == 0:
        print("No valid predictions. Unable to plot ROC/PR curves.")
        return

    y_true_bin = label_binarize(y_true_f, classes=range(NUM_CLASSES))
    y_pred_bin = label_binarize(y_pred_f, classes=range(NUM_CLASSES))

    plt.figure(figsize=(6, 6))
    for i, cls_name in enumerate(CLASSES):
        try:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
            plt.plot(fpr, tpr, label=f"{cls_name}")
        except ValueError:
            pass
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("All Classes ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.savefig(f"{output_prefix}_all_classes_roc_curve.png")
    plt.close()
    print(f"Saved: {output_prefix}_all_classes_roc_curve.png")

    plt.figure(figsize=(6, 6))
    for i, cls_name in enumerate(CLASSES):
        try:
            precision_i, recall_i, _ = precision_recall_curve(y_true_bin[:, i], y_pred_bin[:, i])
            plt.plot(recall_i, precision_i, label=f"{cls_name}")
        except ValueError:
            pass
    plt.title("All Classes PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower right")
    plt.savefig(f"{output_prefix}_all_classes_pr_curve.png")
    plt.close()
    print(f"Saved: {output_prefix}_all_classes_pr_curve.png")

    cm = confusion_matrix(y_true_f, y_pred_f, labels=range(NUM_CLASSES))
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(NUM_CLASSES)
    plt.xticks(tick_marks, CLASSES, rotation=90)
    plt.yticks(tick_marks, CLASSES)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f"{output_prefix}_confusion_matrix.png")
    plt.close()
    print(f"Saved: {output_prefix}_confusion_matrix.png")

    precision_arr = precision_score(y_true_f, y_pred_f, labels=range(NUM_CLASSES), average=None, zero_division=0)
    recall_arr = recall_score(y_true_f, y_pred_f, labels=range(NUM_CLASSES), average=None, zero_division=0)
    specificity_arr = []
    f1_arr = []
    for i in range(NUM_CLASSES):
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        TN = np.sum(cm) - (TP + FP + FN)
        f1_i = 2*TP / (2*TP + FP + FN) if (2*TP + FP + FN) > 0 else 0.0
        f1_arr.append(f1_i)
        denom_spec = (TN + FP)
        specificity_arr.append(TN / denom_spec if denom_spec > 0 else 0.0)
        
    macro_precision = np.mean(precision_arr)
    macro_recall = np.mean(recall_arr)
    macro_specificity = np.mean(specificity_arr)
    macro_f1 = np.mean(f1_arr)

    print("\n=== Per-Class Detailed Metrics ===")
    print(f"{'Class Name':30s} | Precision | Recall(Sens) | Specificity | F1-score")
    print("-" * 85)
    for i, cls_name in enumerate(CLASSES):
        print(f"{cls_name:30s} | {precision_arr[i]:9.3f} | {recall_arr[i]:11.3f} | {specificity_arr[i]:10.3f} | {f1_arr[i]:8.3f}")
    print()

    import csv
    with open(f"{output_prefix}_per_class_metrics.csv", "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Class Name", "Precision", "Recall(Sens)", "Specificity", "F1-score"])
        for i, cls_name in enumerate(CLASSES):
            writer.writerow([
                cls_name,
                round(precision_arr[i], 4),
                round(recall_arr[i], 4),
                round(specificity_arr[i], 4),
                round(f1_arr[i], 4),
            ])
        writer.writerow([])
        writer.writerow([
            "(Macro Average)",
            round(macro_precision, 4),
            round(macro_recall, 4),
            round(macro_specificity, 4),
            round(macro_f1, 4)
        ])
    print(f"Saved CSV: {output_prefix}_per_class_metrics.csv")

    accuracy = accuracy_score(y_true_f, y_pred_f)
    try:
        macro_roc_auc = roc_auc_score(y_true_bin, y_pred_bin, average="macro", multi_class="ovr")
    except ValueError:
        macro_roc_auc = 0.0
    pr_aucs = []
    for i in range(NUM_CLASSES):
        try:
            precision_i, recall_i, _ = precision_recall_curve(y_true_bin[:, i], y_pred_bin[:, i])
            if len(precision_i) > 1 and len(recall_i) > 1:
                pr_aucs.append(auc(recall_i, precision_i))
        except ValueError:
            pass
    macro_pr_auc = np.mean(pr_aucs) if pr_aucs else 0.0

    print("=== Macro Average Metrics ===")
    print(f"Accuracy           : {accuracy:.3f}")
    print(f"Macro Precision    : {macro_precision:.3f}")
    print(f"Macro Recall       : {macro_recall:.3f}")
    print(f"Macro Specificity  : {macro_specificity:.3f}")
    print(f"Macro F1           : {macro_f1:.3f}")
    print(f"Macro ROC AUC      : {macro_roc_auc:.3f}")
    print(f"Macro PR AUC       : {macro_pr_auc:.3f}")
    print("=====================================\n")

def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    model_name = "Salesforce/instructblip-flan-t5-xl"

    global processor
    processor = InstructBlipProcessor.from_pretrained(model_name, use_fast=True)
    model = InstructBlipForConditionalGeneration.from_pretrained(model_name)

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

    processor.image_processor.size = {"height": 224, "width": 224}
    processor.image_processor.crop_size = {"height": 224, "width": 224}

    model.to(device)
    model.config.use_cache = False

    data_file = "/Users/iinuma/Desktop/face_classification_K/annotations/final.json"  
    full_dataset = FaceStainMultiModalDataset(data_file)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, eval_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Eval dataset:  {len(eval_dataset)} samples")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: instructblip_collate_fn(b, processor)
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: instructblip_collate_fn(b, processor)
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir="./instructblip_finetuned_no_image_token",
        num_train_epochs=3,
        per_device_train_batch_size=1,
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

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=lambda b: instructblip_collate_fn(b, processor),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    final_eval_metrics = trainer.evaluate()
    print("\n=== Final Evaluation (trainer.evaluate) ===")
    print(final_eval_metrics)

    analyze_final_results(trainer, eval_dataset, output_prefix="final_eval")

    train_loss = []
    eval_loss = []
    steps = []
    for record in trainer.state.log_history:
        if "loss" in record and "step" in record:
            train_loss.append(record["loss"])
            steps.append(record["step"])
        if "eval_loss" in record:
            eval_loss.append((record.get("step", None), record["eval_loss"]))

    if len(train_loss) > 0:
        plt.figure()
        plt.plot(steps, train_loss, label="train_loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training Loss over Steps")
        plt.legend()
        plt.savefig("training_loss_curve.png")
        plt.close()
        print("Saved training_loss_curve.png")

    if len(eval_loss) > 0:
        plt.figure()
        eval_steps = [x[0] for x in eval_loss if x[0] is not None]
        eval_values = [x[1] for x in eval_loss]
        plt.plot(eval_steps, eval_values, label="eval_loss", color="orange")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Evaluation Loss over Steps")
        plt.legend()
        plt.savefig("evaluation_loss_curve.png")
        plt.close()
        print("Saved evaluation_loss_curve.png")

    trainer.save_model("./instructblip_finetuned_no_image_token")
    print("Fine-tuned model saved at: ./instructblip_finetuned_no_image_token")

if __name__ == "__main__":
    main()