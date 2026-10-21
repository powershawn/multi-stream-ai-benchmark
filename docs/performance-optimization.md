# 六模型推論瓶頸與 CUDA Graph 優化

測試條件維持：同一 `video.mp4`、六個獨立 `yolo11m.pt` 模型、FP16、640 輸入尺寸（此影片實際 tensor 為 1×3×384×640）、每路 15 FPS、每張畫面完成六模型與全部後處理。沒有降低模型數、影像尺寸、精度或驗收門檻。

## 為什麼 GPU 只有 36.9%，仍達不到 15 FPS？

瓶頸主要在 CPU 提交大量小型 CUDA 運算及高階 predict 包裝，而非影片解碼。每張畫面呼叫六次高階 `predict()`，也重複執行六次相同的前處理。套件的 Profile 計時器會在各階段反覆同步 CUDA，另外還有原程式逐模型的同步。

GPU 使用率是採樣區間內裝置忙碌的比例，CPU 尚未把下一批工作送進 GPU 時，GPU 就會等待。CPU 總使用率約 7% 也不能排除這個瓶頸：這台 Docker VM 有 16 個邏輯 CPU，單一核心吃滿只占總量的 6.25%。實際追蹤中 CPU 花了大量時間在 `cudaLaunchKernel` 與 cuDNN 提交路徑。

## 同一程序、同一六模型的量測證據

未啟用 profiler 時，以固定畫面重複量測，時間為完整批次 wall time：

| 路徑／階段 | 平均時間 |
|---|---:|
| 原版六次高階 predict | 109.09 ms |
| 只做一次共同前處理 | 1.10 ms |
| 一般 eager 執行六模型前向運算 | 88.37 ms |
| 六模型 NMS（含輸出複製） | 5.83 ms |
| 只共享前處理、仍 eager 執行 | 95.56 ms |
| 六模型 CUDA Graph 前向運算 | 32.14 ms |
| CUDA Graph＋共同前處理＋六次 NMS | 39.59 ms |
| 影片解碼／取樣 | 3.46 ms |

這些是分開進行的微型測試，不能把列值相加或將其 FPS 當作長時間驗收結果。六模型前向運算從 88.37 ms 降至 32.14 ms，約為原本的 36%。微型 Graph 路徑的 39.59 ms 包含 NMS，但不包含高階 Results 包裝與原圖座標還原；新版正式路徑仍完整還原原圖座標，以偵測 tensor 保留結果而不建立高階 Results 包裝。正式單路測試的完整批次平均時間為 45.73 ms，包含六模型、NMS、座標還原與完成同步。另以 batch 2 量測整批約 72 ms，顯示增加批次不保證滿足每路 15 FPS，因此此次正式驗收仍採 batch 1。

另外執行的 profiler 追蹤，每張畫面有約 3,348 次 `cudaLaunchKernel` 類 API 呼叫與約 72 次同步呼叫。Profiler 本身增加開銷，這份追蹤用於確認呼叫種類與次數，速度比較採上表未啟用 profiler 的 wall time。

原始證據：`results/diagnosis/microbench.json`、`original_profiler.txt`、`original_trace.json`；批次對照位於 `results/diagnosis-batch2/microbench.json`。

## 改動方式

1. 相同取樣畫面的 letterbox、BGR→RGB、正規化、CPU→GPU 複製共用一次。
2. 將六個獨立模型的完整前向運算擷取成 CUDA Graph。一次重播仍會執行全部六模型的計算，不重用前一張畫面的預測、不共用六份模型的權重儲存區。
3. 六份預測各自執行相同的 NMS（conf=0.25、IoU=0.7、max_det=300）及原圖座標還原。
4. 全部六模型與後處理結束後才做一次顯式 CUDA 完成同步，計入完整影格。NMS 必要的內部同步仍保留。
5. 在正式量測前準備所需批次形狀與靜態緩衝區。不同批次的 YOLO anchors 保留強引用，避免記憶體被重新配置後造成錯誤輸出。

實作主要在 `benchmark/cuda_graph.py` 與 `benchmark/yolo.py`，用 `--execution predict` 可選回原版，預設為 `cuda_graph`。映像版本更新為 `multi-stream-ai-benchmark:1.1`；其他電腦不需新增套件，Graph 由相同 `.pt` 權重在目標 GPU 上建立。

這種減少 CPU 提交開銷的用途與 [PyTorch CUDA Graph 官方說明](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs) 一致。

## 沒有省略六模型的驗證

- 三項 CUDA 硬體測試驗證六個不同參數的模型都會產生正確的新輸出、batch 1／2／1 切換安全、未準備的形狀拒絕執行，以及六次引用同一個模型物件會被拒絕。
- 在實際 GPU 容器內，25 項測試全部執行並通過；Docker build 沒有 GPU 時，三項硬體測試會明確標記跳過。
- 取影片 0、3、30、60、88 秒畫面，混合 batch 1／2 並返回最初畫面，六個模型各自比較 7 張輸入，共 42 次模型影像比較。所有最終偵測框 tensor（座標、信心值、類別）都與原 predict 流程逐值完全相同。
- 結果保存在 `results/diagnosis/output_equivalence.json`。此一致性檢查針對上述影片樣本，不是另一次全資料集準確率評估。

## 正式量測與解讀

本次完整搜尋與確認輸出在 `results/20261021-080331-c641ee/`。每路必須達成相同 FPS、丟幀、延遲、記憶體與視窗穩定度條件，最高通過路數再做 3 次各 300 秒確認。

第一個 60 秒單路階段已完成 900／900 個完整六模型影格：15.00 FPS、零丟幀、p95 64.86 ms，GPU 平均 49.52%，VRAM 峰值 1595.86 MiB。每一模型的實際評估數均為 900。

到達 15 FPS 後，流程會等待下一個取樣時刻，因此通過驗收時不需要 GPU 使用率達到 100%。攝影機支援路數必須看完整搜尋與長時間確認結果，不能只用 GPU 百分比或短期峰值換算。

Graph 模式只記錄整組六模型的 CUDA event 時間，不捏造各模型的 wall time。因此 `frames.csv` 的獨立模型毫秒欄位為空；完整端到端時間及 `performance_breakdown` 的群組時間仍有量測。CPU wall time 與 GPU 時間區間重疊，不能直接相加。

## 完整重測結論：最大穩定 1 路

`results/20261021-080331-c641ee/summary.json` 最終為 `status=confirmed`、`max_stable_streams=1`、`confirmed=true`、`at_limit=false`。1 路通過，2 路失敗，因此不繼續增加至 4／8／16 路。三輪確認均以原本的 300 秒、15 FPS 及驗收門檻執行，沒有縮短測試或放寬條件。

| 階段 | 路數 | 每路 FPS | 每路丟幀 | 最差 p95 延遲 | GPU 平均 | 驗收 |
|---|---:|---:|---:|---:|---:|---|
| 搜尋，60 秒 | 1 | 15.000 | 0% | 64.86 ms | 49.52% | 通過 |
| 搜尋，60 秒 | 2 | 10.650 | 29% | 112.63 ms | 72.70% | 未通過 |
| 確認 1，300 秒 | 1 | 15.000 | 0% | 56.64 ms | 51.90% | 通過 |
| 確認 2，300 秒 | 1 | 14.993 | 0.0444% | 59.66 ms | 51.72% | 通過 |
| 確認 3，300 秒 | 1 | 15.000 | 0% | 61.71 ms | 52.65% | 通過 |

三輪確認合計完成 13,498／13,500 張完整影格，整體丟幀率 0.0148%。每個模型各自完成 13,498 張影格評估，總計 80,988 次已完成的模型影格評估；沒有把只完成部分模型的畫面計入完成數。

| 確認輪次 | GPU VRAM 採樣峰值 | CPU 平均 | Linux VM RAM 採樣峰值 |
|---|---:|---:|---:|
| 1 | 1595.99 MiB | 5.28% | 2401.02 MiB（17.2%） |
| 2 | 1616.74 MiB | 5.45% | 2336.85 MiB（16.8%） |
| 3 | 1429.08 MiB | 5.48% | 2323.49 MiB（16.7%） |

為何 2 路仍未達標：每路 15 FPS 表示單路每張畫面的時間預算為 66.67 ms；2 路總共需每秒完成 30 張完整六模型影格，平均服務時間必須不超過 33.33 ms。兩路測試中，整組前向 CUDA event 區間平均約 32.85 ms，完整批次 wall time 平均約 46.85 ms，實際只完成總計 21.30 FPS，因此平均每路 10.65 FPS。剩餘前後處理與提交開銷仍影響容量，不能拿剩餘 GPU 使用率直接換算路數。此結果是目前 Python／PyTorch CUDA Graph 流程的驗收容量，不是所有推論引擎下的 GPU 硬體絕對上限。

交付映像為 `multi-stream-ai-benchmark:1.1`，ID：`sha256:bf3ae4bf3b2fb520099cee6bfafda32cbfe48f1a8dc6c65054928191a7cea755`。影片與權重的 SHA-256 已核對與 1.0 基準相同。

重跑新版完整搜尋：

```sh
docker compose run --rm benchmark run
```

原版流程對照：

```sh
docker compose run --rm benchmark smoke --duration 60 --execution predict
```

完整每路數據、全部資源採樣與原始影格時間位於該次 `summary.json`、`summary.csv`，及各階段的 `frames.csv`、`resources.csv`。1.0 的原始基準仍保留於 `results/20261021-072606-ee8586/` 與 `docs/verification.md`。
