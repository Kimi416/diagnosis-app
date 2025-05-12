#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
医療診断アプリ - InstructBLIP + ChatGPT API(v1) を使用した皮膚疾患診断
1) ローカルの Fine-tuned InstructBLIP で一次診断
2) OpenAI v1 API client で最終診断を取得
"""

import os
import torch
from PIL import Image
import streamlit as st
import openai
from openai import OpenAI
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

# ChatGPT API キー取得（環境変数）
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("🚨 環境変数 OPENAI_API_KEY が設定されていません")
    st.stop()

# OpenAI v1 client 初期化
client = OpenAI(api_key=api_key)

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
    LOCAL_DIR = "./instructblip_finetuned_no_image_token"
    if os.path.isdir(LOCAL_DIR):
        st.info(f"🔄 ファインチューン済みモデル読み込み: {LOCAL_DIR}")
        model     = InstructBlipForConditionalGeneration.from_pretrained(LOCAL_DIR)
        processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl", use_fast=False)
    else:
        st.warning(f"⚠️ モデルフォルダ未発見: {LOCAL_DIR} -> デフォルトロード")
        processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl", use_fast=False)
        model     = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl")

    # デバイス
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


def local_inference(model, processor, device, image, symptoms, exam):
    """一次診断：画像＋テキスト -> ラベル＆説明"""
    img = image.convert("RGB")
    prompt = f"USER: この患者さんの写真を診断してください。\n症状経過: {symptoms}\n触診結果: {exam}\nASSISTANT:"
    inputs = processor(images=img, text=prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k,v in inputs.items()}
    with torch.no_grad():
        out_ids = model.generate(**inputs, num_beams=4, early_stopping=True, max_length=128)
    text = processor.tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]
    idx  = get_class_index_from_generated(text)
    if idx>=0:
        label = CLASSES[idx]
        expl  = DISEASE_EXPLANATIONS[label]
    else:
        label = "不明"
        expl  = "モデルが疾患を特定できませんでした。専門医へ。"
    return label, expl, text


def chatgpt_refine(label, expl, symptoms, exam):
    """一次診断をもとに ChatGPT API(v1) で最終診断取得"""
    system = "あなたは皮膚科医です。一次診断と情報から最終診断を述べてください。"
    user   = f"一次診断: {label}\n診断理由: {expl}\n症状経過: {symptoms}\n触診結果: {exam}"
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
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
        label, expl, raw = local_inference(model,processor,device,img,symptoms,exam)
        final = chatgpt_refine(label,expl,symptoms,exam)

        with col2:
            st.subheader("一次診断 (InstructBLIP)")
            st.markdown(f"**ラベル**: {label}")
            st.markdown(f"**説明**: {expl}")
            st.code(raw)
            st.subheader("最終診断 (ChatGPT)")
            st.markdown(final)
            st.image(img, caption="入力画像", use_column_width=True)

if __name__ == "__main__":
    main()
