#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
医療診断アプリ - InstructBLIPモデルを使用した皮膚疾患診断

写真情報、病気の経過、触診結果を入力して診断名を提示するWebアプリケーション
"""

import os
import torch
import numpy as np
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
    生成テキストからクラスを特定する簡易関数。
    CLASSES の疾患名が部分一致すれば、そのクラスを返す (先にマッチした方優先)。
    見つからなければ -1 を返す (未知クラス扱い)。
    """
    gen_lower = generated_text.lower()
    for i, cls_name in enumerate(CLASSES):
        if cls_name.lower() in gen_lower:
            return i
    return -1

def load_model():
    """
    InstructBLIPモデルを読み込みます。
    """
    model_path = "instructblip_finetuned_no_image_token"  # 学習済みモデルのパス
    
    # モデルパスが存在しない場合はHuggingFaceのデフォルトモデルを使用
    if not os.path.exists(model_path):
        model_path = "Salesforce/instructblip-flan-t5-xl"
        st.warning("学習済みモデルが見つかりません。デフォルトのInstructBLIPモデルを使用します。")
    
    # GPUが利用可能な場合はGPUを、Macの場合はMPSを、それ以外の場合はCPUを使用
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    processor = InstructBlipProcessor.from_pretrained(model_path, use_fast=True)
    model = InstructBlipForConditionalGeneration.from_pretrained(model_path)
    
    # 画像処理用の設定
    processor.image_processor.size = {"height": 224, "width": 224}
    processor.image_processor.crop_size = {"height": 224, "width": 224}
    
    model.to(device)
    model.eval()  # 評価モードに設定
    
    return model, processor, device

def diagnose(model, processor, device, image, symptoms, examination):
    """
    写真、症状の経過、触診結果を基に診断を行います。
    """
    # 画像をRGBに変換
    image = image.convert("RGB")
    
    # プロンプトの作成（症状と触診結果を含む）
    prompt = f"USER: この患者さんの写真を見て、診断を行ってください。\n症状の経過: {symptoms}\n触診結果: {examination}\nASSISTANT:"
    
    # モデル入力の準備
    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 生成
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=512,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
        )
    
    # 結果のデコード
    generated_text = processor.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    
    # 診断結果のクラスを取得
    diagnosis_idx = get_class_index_from_generated(generated_text)
    if diagnosis_idx >= 0:
        diagnosis = CLASSES[diagnosis_idx]
        explanation = DISEASE_EXPLANATIONS.get(diagnosis, "詳細情報なし")
    else:
        diagnosis = "不明"
        explanation = "モデルが疾患を特定できませんでした。より詳細な情報を提供するか、専門医に相談してください。"
    
    return generated_text, diagnosis, explanation

def main():
    """
    StreamlitアプリのメインUI
    """
    st.set_page_config(page_title="皮膚疾患診断アプリ", page_icon="🏥", layout="wide")
    
    st.title("🏥 皮膚疾患診断アプリ")
    st.subheader("写真、症状の経過、触診結果から診断を行います")
    
    # サイドバーに説明を追加
    with st.sidebar:
        st.header("このアプリについて")
        st.info(
            "このアプリはInstructBLIPモデルを使用して皮膚疾患の診断を行います。"
            "写真、症状の経過、触診結果を入力することで、可能性のある診断名を提示します。"
            "\n\n**注意**: これはあくまで参考情報であり、最終的な診断は必ず医療専門家に相談してください。"
        )
        
        st.header("診断可能な疾患")
        for cls in CLASSES:
            st.write(f"- **{cls}**: {DISEASE_EXPLANATIONS.get(cls, '情報なし')}")
    
    # 入力エリア
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("患者情報の入力")
        
        # 写真のアップロード
        uploaded_file = st.file_uploader("疾患部位の写真をアップロード", type=["jpg", "jpeg", "png"])
        
        # 症状の経過
        symptoms = st.text_area("症状の経過", 
                             placeholder="例: 2ヶ月前から徐々に大きくなってきた。痒みや痛みはない。",
                             height=150)
        
        # 触診結果
        examination = st.text_area("触診結果", 
                               placeholder="例: 弾性硬、可動性あり、圧痛なし。",
                               height=100)
        
        # 診断ボタン
        diagnose_button = st.button("診断を行う", type="primary", disabled=uploaded_file is None)
    
    # 診断が実行された場合
    if diagnose_button and uploaded_file is not None:
        with st.spinner("診断中...モデルが分析しています..."):
            # モデルのロード（初回のみ）
            if "model" not in st.session_state:
                model, processor, device = load_model()
                st.session_state.model = model
                st.session_state.processor = processor
                st.session_state.device = device
            else:
                model = st.session_state.model
                processor = st.session_state.processor
                device = st.session_state.device
            
            # 画像の読み込み
            image = Image.open(uploaded_file)
            
            # 診断の実行
            generated_text, diagnosis, explanation = diagnose(
                model, processor, device, image, symptoms, examination
            )
        
        # 診断結果の表示
        with col2:
            st.header("診断結果")
            
            # 画像の表示
            st.image(image, caption="アップロードされた画像", use_column_width=True)
            
            # 診断名と説明
            st.subheader(f"診断: {diagnosis}")
            st.markdown(f"**説明**: {explanation}")
            
            # 詳細分析
            st.subheader("詳細分析")
            st.markdown(f"```{generated_text}```")
            
            # 免責事項
            st.warning(
                "注意: この診断結果はAIによる予測であり、医療アドバイスの代わりにはなりません。"
                "正確な診断と治療については、必ず医療専門家に相談してください。"
            )

if __name__ == "__main__":
    main()
