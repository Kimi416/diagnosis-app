# 皮膚疾患診断アプリ

このアプリケーションは、画像と症状の説明に基づいて皮膚疾患の診断を行います。InstructBLIPモデルを使用して、視覚情報とテキスト情報の両方から診断を行います。

## 機能

- 皮膚疾患の画像をアップロード
- 症状や検査所見をテキストで入力
- AIモデルによる診断結果の表示
- 学習済みモデルのカスタマイズ

## 必要条件

- Python 3.9以上
- pip（Pythonパッケージマネージャー）
- インターネット接続（初回実行時にモデルをダウンロードするため）

## インストール方法

1. リポジトリをクローンするか、ファイルをダウンロードします。
2. 仮想環境を作成して有効化します：

```bash
python -m venv venv
source venv/bin/activate  # Linuxまたは Mac
# または
venv\Scripts\activate  # Windows
```

3. 必要なパッケージをインストールします：

```bash
pip install -r requirements.txt
```

## 使用方法

### アプリの起動

次のコマンドでアプリを起動します：

```bash
./run_app.sh
```

または手動で：

```bash
source venv/bin/activate
streamlit run app.py
```

ブラウザで http://localhost:8501 を開き、アプリケーションにアクセスします。

### モデルのトレーニング

カスタムモデルをトレーニングするには、次のコマンドを実行します：

```bash
source venv/bin/activate
python train_instructblip_improved.py --json_path=/path/to/your/data.json
```

オプションのパラメータ：

- `--json_path`: 学習データのJSONファイルへのパス（必須）
- `--output_dir`: 学習済みモデルの保存先ディレクトリ（デフォルト: `./instructblip_finetuned_no_image_token`）
- `--epochs`: 学習エポック数（デフォルト: 3）
- `--batch_size`: バッチサイズ（デフォルト: 1）

### トレーニングデータの形式

トレーニングデータは次の形式のJSONファイルである必要があります：

```json
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
```

### 学習済みモデルの使用

1. アプリのサイドバーで「モデル設定」セクションを見つけます。
2. トレーニング済みモデルの絶対パスを入力します（例: `/Users/username/app/instructblip_finetuned_no_image_token`）。
3. 「モデル設定を適用」ボタンをクリックします。
4. 画像をアップロードし、症状を入力して「診断を行う」ボタンをクリックします。

## 診断可能な疾患

以下の皮膚疾患を診断することができます：

- ADM
- Basal cell carcinoma
- Ephelis
- Malignant melanoma
- Melasma
- Nevus
- Seborrheic keratosis
- Solar lentigo

## トラブルシューティング

- **モデルが見つからないエラー**: カスタムモデルパスが正しいことを確認してください。パスは絶対パスである必要があります。
- **画像処理エラー**: サポートされている画像形式（JPG、PNG）を使用しているか確認してください。
- **メモリエラー**: 大きな画像を使用している場合は、より小さなサイズにリサイズしてから試してください。

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 問い合わせ

ご質問やフィードバックがございましたら、[イシュートラッカー](https://github.com/yourusername/app/issues)にてご連絡ください。 