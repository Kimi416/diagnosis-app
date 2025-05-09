#!/bin/bash

# 皮膚疾患診断アプリケーション起動スクリプト

echo "皮膚疾患診断アプリケーションを起動しています..."

# 仮想環境があれば有効化（オプション）
# source venv/bin/activate

# 必要なパッケージが入っているか確認
pip list | grep streamlit > /dev/null
if [ $? -ne 0 ]; then
    echo "Streamlitが見つかりません。必要なパッケージをインストールしています..."
    pip install -r requirements.txt
fi

# アプリケーションを起動
streamlit run app.py

# 終了メッセージ
echo "アプリケーションを終了しました。" 