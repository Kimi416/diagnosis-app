#!/bin/bash

# 皮膚疾患診断アプリケーション起動スクリプト

echo "皮膚疾患診断アプリケーションを起動しています..."

# 仮想環境の有効化
if [ -d "venv" ]; then
    echo "仮想環境を有効化しています..."
    source venv/bin/activate
else
    echo "仮想環境が見つかりません。新しい仮想環境を作成しています..."
    python3 -m venv venv
    source venv/bin/activate
fi

# 必要なパッケージが入っているか確認
python -m pip list | grep streamlit > /dev/null
if [ $? -ne 0 ]; then
    echo "Streamlitが見つかりません。必要なパッケージをインストールしています..."
    python -m pip install -r requirements.txt
fi

# アプリケーションを起動（ポート8502を使用）
python -m streamlit run app.py --server.port 8502

# 終了メッセージ
echo "アプリケーションを終了しました。" 