#!/usr/bin/env python
# -*- coding: utf-8 -*-

# .env から環境変数をロード
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import os
import torch
from PIL import Image
import streamlit as st
from openai import OpenAI
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

# ────────────────────────────────────────────
# 環境変数の取得
API_KEY    = os.getenv("OPENAI_API_KEY")
MODEL_DIR  = os.getenv("FINETUNED_MODEL_DIR", "./instructblip_finetuned_no_image_token")
CHAT_MODEL = os.getenv("CHATGPT_MODEL_NAME", "gpt-4o-mini")
# ────────────────────────────────────────────

# APIキー未設定時は停止
if not API_KEY:
    st.error("🚨 環境変数 OPENAI_API_KEY が設定されていません")
    st.stop()

# OpenAI v1 client 初期化
client = OpenAI(api_key=API_KEY)

# 診断可能な疾患クラス
CLASSES = [
    "ADM", "Basal_cell_carcinoma", "Ephelis", "Malignant_melanoma",
    "Melasma", "Nevus", "Seborrheic_keratosis", "Solar_lentigo"
]
# 疾患説明マップ
DISEASE_EXPLANATIONS = {
    "ADM": "異型黒子症 - 色素性病変で、メラノーマの前駆病変となる可能性があります。",
    "Basal_cell_carcinoma": "基底細胞癌 - 最も一般的な皮膚がんで、ゆっくり成長し周囲を侵食します。",
    "Ephelis": "そばかす - 遺伝と紫外線により生じる褐色斑。",
    "Malignant_melanoma": "悪性黒色腫 - 最も危険な皮膚がんで早期診断が重要。",
    "Melasma": "肝斑 - ホルモン変化や紫外線誘発の茶褐色沈着。",
    "Nevus": "母斑/ほくろ - 良性色素性病変。",
    "Seborrheic_keratosis": "脂漏性角化症 - 良性増殖でワックス状。",
    "Solar_lentigo": "日光性黒子 - 紫外線曝露による褐色斑。"
}


def get_class_index_from_generated(text: str) -> int:
    """生成テキスト部分一致で CLASSES の index を返す"""
    low = text.lower()
    for i, cls in enumerate(CLASSES):
        if cls.lower() in low:
            return i
    return -1


def load_local_model():
    """ファインチューンモデル優先ロード、なければデフォルト"""
    LOCAL_DIR = MODEL_DIR
    if os.path.isdir(LOCAL_DIR):
        st.info(f"🔄 ファインチューン済みモデル読み込み: {LOCAL_DIR}")
        model     = InstructBlipForConditionalGeneration.from_pretrained(LOCAL_DIR)
        processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl", use_fast=False)
    else:
        st.warning(f"⚠️ モデルフォルダ未発見: {LOCAL_DIR} -> デフォルトロード")
        processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl", use_fast=False)
        model     = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl")

    # デバイス選択
    if torch.cuda.is_available():
        device = torch.device("cuda"); st.sidebar.success("GPU 使用中")
    elif torch.backends.mps.is_available():
        device = torch.device("mps");  st.sidebar.success("MPS 使用中")
    else:
        device = torch.device("cpu");  st.sidebar.info("CPU 使用中")

    # 前処理サイズ
    processor.image_processor.size      = {"height":224, "width":224}
    processor.image_processor.crop_size = {"height":224, "width":224}

    model.to(device)
    model.eval()
    return model, processor, device


def diagnose_together(model, processor, device, image, symptoms, exam):
    """
    InstructBLIP の生の出力と臨床情報を合わせて
    ChatGPT に一緒に考えてもらい最終診断を返す
    """
    # --- InstructBLIP 推論 ---
    img = image.convert("RGB")
    prompt = (
        f"この患者さんの写真を診断してください。\n"
        f"症状経過: {symptoms}\n"
        f"触診結果: {exam}\n"
        f"→ まずは InstructBLIP モデルの判断を生の文章で出力してください。"
    )
    inputs = processor(images=img, text=prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out_ids = model.generate(**inputs, num_beams=4, early_stopping=True, max_length=128)
    raw_text = processor.tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]

    # --- ChatGPT 連携 ---
    system = "あなたは熟練の皮膚科医です。"
    user_msg = (
        f"以下は InstructBLIP モデルの出力です。\n"
        f"{raw_text}\n\n"
        f"これと、以下の臨床情報を合わせて一緒に考え、最終的な診断とその理由を一つの文章で示してください。\n\n"
        f"【症状経過】: {symptoms}\n"
        f"【触診結果】: {exam}"
    )
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.5,
        max_tokens=256,
    )
    return resp.choices[0].message.content.strip()


def main():
    st.set_page_config(page_title="皮膚疾患診断アプリ", page_icon="🏥", layout="wide")
    st.title("🏥 皮膚疾患診断 (InstructBLIP + ChatGPT)")

    # サイドバー
    with st.sidebar:
        st.header("このアプリについて")
        st.info("一次診断に Fine-tuned InstructBLIP、最終診断に ChatGPT API(v1) を使用します。結果は参考情報です。")
        st.header("診断可能疾患")
        for cls in CLASSES:
            st.write(f"- **{cls}**: {DISEASE_EXPLANATIONS[cls]}")

    col1, col2 = st.columns(2)
    with col1:
        img_file = st.file_uploader("写真アップロード", type=["jpg","png","jpeg"])
        symptoms = st.text_area("症状の経過", height=120)
        exam     = st.text_area("触診結果", height=80)
        btn      = st.button("診断開始", type="primary", disabled=img_file is None)

    if btn and img_file:
        img = Image.open(img_file)
        model, processor, device = load_local_model()
        # InstructBLIP の出力と臨床情報をまとめて一緒に診断
        diagnosis = diagnose_together(model, processor, device, img, symptoms, exam)

        with col2:
            st.subheader("診断結果")
            st.markdown(diagnosis)
            st.image(img, caption="入力画像", use_column_width=True)

if __name__ == "__main__":
    main()
