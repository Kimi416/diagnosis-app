#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
医療診断アプリ - InstructBLIPモデルを使用した皮膚疾患診断
写真情報、病気の経過、触診結果を入力して診断名を提示するWebアプリケーション
"""

import os
import torch
from PIL import Image
import streamlit as st
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

# 診断可能な疾患クラス
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

# 疾患名と日本語での説明
DISEASE_EXPLANATIONS = {
    "ADM": "異型黒子症 - 色素性病変で、メラノーマの前駆病変となる可能性があります。",
    "Basal_cell_carcinoma": "基底細胞癌 - 最も一般的な皮膚がんの一種で、通常はゆっくりと成長し、周囲の組織を侵食します。",
    "Ephelis": "そばかす - 遺伝的要因と太陽光への露出により発生する小さな褐色の斑点です。",
    "Malignant_melanoma": "悪性黒色腫 - 最も危険な皮膚がんの一種で、早期発見と治療が重要です。",
    "Melasma": "肝斑 - ホルモンの変化や太陽光への露出により顔に現れる褐色の色素沈着です。",
    "Nevus": "母斑/ほくろ - 皮膚に発生する色素性の良性病変です。",
    "Seborrheic_keratosis": "脂漏性角化症 - 年齢とともに発生する良性の皮膚成長物で、表面は通常ワックス状または油性に見えます。",
    "Solar_lentigo": "日光性黒子 - 長期の太陽光露出により発生する褐色の斑点で、「老人性色素斑」とも呼ばれます。"
}


def get_class_index_from_generated(generated_text: str) -> int:
    """
    生成テキストから CLASSES と部分一致でクラスを取得
    見つからなければ -1 を返す
    """
    gen_lower = generated_text.lower()
    for i, cls_name in enumerate(CLASSES):
        if cls_name.lower() in gen_lower:
            return i
    return -1


def load_model():
    LOCAL_MODEL_DIR = "./instructblip_finetuned_no_image_token"
    # モデルはローカル、プロセッサはデフォルト
    if os.path.isdir(LOCAL_MODEL_DIR):
        st.info(f"🔄 ファインチューン済みモデルを読み込んでいます: {LOCAL_MODEL_DIR}")
        model     = InstructBlipForConditionalGeneration.from_pretrained(LOCAL_MODEL_DIR)
        processor = InstructBlipProcessor.from_pretrained(
            "Salesforce/instructblip-flan-t5-xl", use_fast=False
        )
    else:
        st.warning(f"⚠️ ローカルモデルが見つかりません ({LOCAL_MODEL_DIR})。デフォルトモデルを読み込みます。")
        processor = InstructBlipProcessor.from_pretrained(
            "Salesforce/instructblip-flan-t5-xl", use_fast=False
        )
        model     = InstructBlipForConditionalGeneration.from_pretrained(
            "Salesforce/instructblip-flan-t5-xl"
        )

    # デバイス選定
    if torch.cuda.is_available():
        device = torch.device("cuda");    st.sidebar.success("GPU (CUDA) を使用中")
    elif torch.backends.mps.is_available():
        device = torch.device("mps");     st.sidebar.success("Apple Silicon (MPS) を使用中")
    else:
        device = torch.device("cpu");     st.sidebar.info("CPU を使用中")

    # 画像前処理サイズ設定
    processor.image_processor.size      = {"height":224,"width":224}
    processor.image_processor.crop_size = {"height":224,"width":224}

    model.to(device)
    model.eval()
    return model, processor, device



def diagnose(model, processor, device, image, symptoms, examination):
    """
    写真、症状経過、触診結果から診断を生成
    """
    image = image.convert("RGB")
    prompt = (
        f"USER: この患者さんの写真を見て、診断を行ってください。"
        f"\n症状の経過: {symptoms}\n触診結果: {examination}\nASSISTANT:"
    )
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            num_beams=4,
            early_stopping=True,
            max_length=128,
        )
    generated = processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

    idx = get_class_index_from_generated(generated)
    if idx >= 0:
        cls = CLASSES[idx]
        explanation = DISEASE_EXPLANATIONS.get(cls, "詳細情報なし")
    else:
        cls = "不明"
        explanation = (
            "モデルが疾患を特定できませんでした。より詳細な情報を提供するか、専門医に相談してください。"
        )
    return generated, cls, explanation


def main():
    st.set_page_config(page_title="皮膚疾患診断アプリ", page_icon="🏥", layout="wide")
    st.title("🏥 皮膚疾患診断アプリ")
    st.subheader("写真、症状の経過、触診結果から診断を行います")

    # サイドバーにアプリ説明と疾患リストのみを表示
    with st.sidebar:
        st.header("このアプリについて")
        st.info(
            "このアプリはInstructBLIPモデルを使用して皮膚疾患の診断を行います。"
            "写真、症状の経過、触診結果を入力することで、診断名を提示します。"
            "\n\n⚠️ 参考情報です。最終的な診断は専門医へ。"
        )
        st.header("診断可能な疾患")
        for cls in CLASSES:
            st.write(f"- **{cls}**: {DISEASE_EXPLANATIONS.get(cls, '')}")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.header("患者情報の入力")
        uploaded_file = st.file_uploader(
            "疾患部位の写真をアップロード", type=["jpg", "jpeg", "png"]
        )
        symptoms = st.text_area(
            "症状の経過",
            placeholder="例: 2ヶ月前から徐々に大きくなってきた。痒みや痛みはない。",
            height=150,
        )
        examination = st.text_area(
            "触診結果",
            placeholder="例: 弾性硬、可動性あり、圧痛なし。",
            height=100,
        )
        do_btn = st.button("診断を行う", type="primary", disabled=uploaded_file is None)

    if do_btn and uploaded_file is not None:
        with st.spinner("診断中…モデルが分析しています…"):
            if "model" not in st.session_state:
                model, processor, device = load_model()
                st.session_state.update({"model": model, "processor": processor, "device": device})
            else:
                model = st.session_state["model"]
                processor = st.session_state["processor"]
                device = st.session_state["device"]

            img = Image.open(uploaded_file)
            gen_txt, diag, expl = diagnose(model, processor, device, img, symptoms, examination)

        with col2:
            st.header("診断結果")
            st.image(img, caption="アップロード画像", use_column_width=True)
            st.subheader(f"診断: {diag}")
            st.markdown(f"**説明**: {expl}")
            st.subheader("詳細分析")
            st.code(gen_txt)
            st.warning("⚠️ 本結果はAI予測です。最終診断・治療は専門医へ。")

if __name__ == "__main__":
    main()
