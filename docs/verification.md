# 2026-10-21 實測與交付紀錄

此文件保留 **1.0 版 predict 流程的基準結果**。1.1 版已整合 CUDA Graph 優化，最新診斷、驗證與重測結果見 `docs/performance-optimization.md`；不要將本文件的 0 路結果當作新版結論。

Python／Docker benchmark 已建置，映像名稱為 `multi-stream-ai-benchmark:1.0`。20 項自動測試已在實際 Docker 環境通過，包含逐路驗收、路數搜尋、確認失敗後降載、影片時間軸與循環、截止計數、缺失資源資料、GPU UUID 對應及 subprocess 報告流程。

## 這台電腦的正式結果

在**每路 15 FPS、每張取樣畫面完成六份獨立 YOLO11-M 推論、batch=1、FP16、imgsz=640**的設定下，單路沒有通過驗收，因此沒有可宣稱通過的正路數。

| 指標 | 60 秒正式量測 |
|---|---:|
| 攝影機／模型 | 1 路 × 6 個獨立 yolo11m.pt |
| 預定分析影格 | 900 |
| 在截止前完成全部六模型的影格 | 495 |
| 完整六模型 FPS | 8.25 |
| 完整端到端延遲 p95 | 197.82 ms |
| 未完成／丟幀比例 | 45.00% |
| 最慢 5 秒區間 FPS | 7.60 |
| GPU 平均／採樣峰值使用率 | 36.87%／42% |
| 整張 GPU VRAM 採樣峰值 | 1326.21 MiB（21.59%） |
| CPU 平均使用率 | 7.13% |
| Linux VM RAM 採樣峰值 | 2573 MiB（18.3%） |
| benchmark 程序 RSS 採樣峰值 | 1861.48 MiB |
| 資源採樣 | 60 筆，無缺失警告 |

六個模型都已成功完成真實影片單張預檢，功能檢查正常。15 秒 smoke 量測為 8.8 FPS，之後的 60 秒正式量測如上。正式搜尋從 1 路開始，因 1 路即未通過 FPS、視窗穩定度與丟幀限制，所以沒有繼續測 2／4／8／16 路，也沒有候選路數可進行 3 × 300 秒長測。

`max_stable_streams=0` 表示「這套測試流程與驗收設定下沒有路數通過」，不代表這張 GPU 在其他推論架構下無法支援攝影機。GPU 並未滿載，這次不能據此斷言 GPU 的硬體上限。批次、執行方式或推論引擎改變後，需要重新量測。CPU／RAM 數據來自 Docker Linux VM，不等於 Windows 工作管理員的整機數值。

## 環境與原始結果

- NVIDIA GeForce RTX 3060 Laptop GPU，6 GiB VRAM，驅動 610.78。
- PyTorch 2.6.0+cu124、CUDA 12.4、Ultralytics 8.3.253、OpenCV 4.10.0。
- 影片 1280 × 720、29.53 FPS、2638 幀，長度約 89.33 秒。
- 六份模型常駐 GPU，各路共用；六份模型依序處理每個推論批次。
- 正式結果：`results/20261021-072606-ee8586/`。
- 單路 smoke：`results/20261021-072428-ea7b75/`。
- 正式原始數據位於該資料夾的 `summary.json`、`summary.csv`，以及階段內的 `frames.csv`、`resources.csv`、`preflight.json`、`stage.log`。
- 交付映像 ID：`sha256:eff6a8da2456848b3dfbc0a58d1712c894b2f3ad707254e8ac21d9ce8f3e3ea9`。

## 重跑與換機

本機已有映像與權重：

```sh
docker compose run --rm benchmark run
```

新機可複製專案後執行 `docker compose build`、`docker compose run --rm benchmark prepare`，再執行同一個 `run` 命令。若要固定完全相同的已建置環境，可照根目錄 README 使用 `docker save`／`docker load`，並一同複製 `compose.yaml`、`config.json`、`models/`、`video.mp4`。

目前程式與實測資料均保留在工作目錄，尚未建立 Git commit。
