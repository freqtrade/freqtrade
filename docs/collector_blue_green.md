# Orderbook Collector Blue-Green Deployment

`tools/ob_collector_ws.py` (Bybit official WebSocket > 1s last > 1m parquet) を Blue-Green 方式で
無停止アップデートするための docker-compose 構成です。ccxtpro は不要で、公式 WebSocket と
REST スナップショットのみを使用します。

## 事前準備

1. 環境変数を `docker/docker-compose-ob-collector-bluegreen.yml` で調整します。
   - `EXCHANGE`, `SYMBOL`, `CATEGORY` (linear/spot)、`DEPTH`
   - `ROOT_DIR` はホスト側 `user_data/featurestore/...` へマウントされます。
2. 必要であれば `docker/requirements-ob-collector.txt` に依存を追加してください。

## イメージのビルド

```powershell
# リポジトリルートで実行
docker compose -f docker/docker-compose-ob-collector-bluegreen.yml build --pull
```

## 新バージョン (green) のローンチ

```powershell
# 先に green 側だけを起動
docker compose -f docker/docker-compose-ob-collector-bluegreen.yml up -d ob_collector_green

# ログで 1 分バッチが出力されているか確認
docker logs -f ob-collector-green
```

`user_data/featurestore/...` に新しい Parquet が出力されていることを確認します。

## Blue からの切り替え

```powershell
# 旧バージョン (blue) の collector を停止
docker stop ob-collector-blue
```

必要になった場合は `docker start ob-collector-blue` で即座にロールバックできます。

## 次回リリースに備えて

切り替え後は blue イメージを新バージョンで再ビルドしておくと、次のリリースでも同じ手順
(先に green → blue 停止) を繰り返せます。

```powershell
docker compose -f docker/docker-compose-ob-collector-bluegreen.yml build ob_collector_blue
```

> 両コンテナは同じ featurestore ディレクトリを共有し、`freqtrade_ext.feature_store.load_orderbook_features`
> が `ts` で重複排除するため、一時的に 2 系統が同時書き込みしてもデータは冪等に扱えます。
