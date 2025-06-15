FROM python:3.12-slim

# 作業ディレクトリ設定
WORKDIR /app

# システムパッケージ更新とPostgreSQLクライアントインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# 依存関係をコピーしてインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY . .

# 環境変数設定
ENV PYTHONUNBUFFERED=1

# コンテナ起動時のコマンド
CMD ["python", "main.py"]