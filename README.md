# 多路攝影機 × 六個 YOLO-M GPU 容量測試

以固定 `video.mp4` 模擬獨立攝影機。**目前每路每秒分析 10 張畫面，每張畫面必須完成六份獨立 `yolo11m.pt` 推論。** 六份模型常駐同一張 GPU，供各路共用；不是每加一路再載入六份模型。預設使用 CUDA Graph 減少 CPU 提交開銷，沒有啟用 TensorRT。1.2 版新增指定路數與中文參數說明；**1.3 版每次測試自動記錄當下 GPU／CPU／RAM 與環境，不固定使用此電腦的規格。**

本機 RTX 3060 Laptop GPU 最新 **1.3 版完整測試在每路 10 FPS 下，確認 1 路通過 3 × 300 秒**，報告為 `results/20261022-060211-722ac4/report.md`。雙路為每路 9.87／9.85 FPS，但丟幀率 1.33%／1.50% 超過 1% 上限，因此未通過，依清單規則停止增加路數。設備規格由本次執行自動擷取。

之前 1.2 版曾在相同 10 FPS 設定下確認 2 路（`results/20261022-051829-1cf59e/`）；此歷史結果不代表每次執行都能通過雙路。兩次結果都保留，未放寬門檻或改寫舊紀錄；差異原因尚未另行診斷，不能僅憑一次差異歸因於硬體記錄功能。之前 15 FPS 設定下為 1 路通過；詳細瓶頸與 CUDA Graph 驗證見 `docs/performance-optimization.md`。不同影片、機器或設定需重新測試。

## 快速開始

先啟動 Docker Desktop（Windows 使用 Linux containers／WSL 2），確認 `nvidia-smi` 正常。Linux 主機需安裝 NVIDIA 驅動與 NVIDIA Container Toolkit。影片已放在專案根目錄。

在本資料夾開啟 PowerShell 或終端機：

```sh
docker compose build
docker compose run --rm benchmark prepare
docker compose run --rm benchmark smoke --duration 15
docker compose run --rm benchmark run
```

- `prepare` 首次下載 YOLO11-M 權重到主機 `models/`；之後不用重複下載。雖然只有一個權重檔，程式會載入六個獨立模型物件。
- `smoke` 先做單路六模型檢查，並產生一份 15 秒測試報告。若驗收失敗會回傳結束碼 1，仍可查閱報告；它不會宣稱已找到最大穩定路數。
- `run` 預設依 `test_streams` 清單測試 1 → 2 → 4 → 8 → 16 路；**第一個未通過就停止清單，不測後面更大的路數，也不補測中間整數。** 例如 4 路失敗，不會測 8／16 或補測 3 路。
- 搜尋每個路數預設暖機 5 秒、正式量測 60 秒。清單結束後，對最高已通過路數連續做 **3 次 × 300 秒**確認；清單模式下任一次長測失敗就結束，不再降低路數重測。
- 若第一個指定路數未通過，就不執行後續路數或長測。清單報告只代表「指定路數中最高已確認值」，不代表未測過的路數也已驗證。
- `test_streams=null` 可選回原版自動搜尋：以 `max_streams` 為上限、倍增後補測中間整數，長測失敗會降低一路重試。

完整測試可能需要約 20 分鐘以上，降低路數後的重測會增加時間。不要同時啟動多份 benchmark。

## 設定參數與測試路數

直接編輯 `config.json`；檔案底部的 **`_help` 已逐一列出每個參數的中文用途、單位與影響**。它是合法 JSON 的說明區塊，不影響測試，結果中只保存實際設定。請不要加入 `//` 或 `#` 註解。

例如維持每路 10 FPS，依序測 1、2、4、8 路：

```json
"target_fps": 10,
"test_streams": [1, 2, 4, 8]
```

僅測指定的 2 路（仍含原本的長測確認）：

```json
"test_streams": [2]
```

路數必須是正整數、嚴格遞增、不重複，不能用空清單。清單模式的路數不受 `max_streams` 限制，也不會自動加測清單以外的路數。

若要原版自動搜尋最大穩定路數：

```json
"test_streams": null,
"max_streams": 16
```

首次升級到 1.3 需重建映像；之後只改 `config.json` 不需重建，因為 Compose 會直接掛載主機設定檔：

```sh
docker compose build
docker compose run --rm benchmark run
```

Windows 也可使用：

```powershell
.\run.ps1 -Mode smoke
.\run.ps1 -Mode run
```

## 每次自動記錄當下測試設備（1.3）

每次 `run` 或 `smoke` 都建立新的硬體快照，寫入該次 `environment.json`、`summary.json` 與 `report.md`，不讀取上一次報告，也不把 RTX 3060、i7 或 32 GiB 等規格寫死。`summary.csv` 另外包含 GPU 名稱／UUID／驅動、CPU 型號與記憶體容量，方便比較不同設備。

| 自動記錄 | 來源與範圍 |
|---|---|
| GPU 型號、UUID、VRAM、Compute Capability、CUDA runtime、驅動 | 推論 worker 實際選用的 CUDA 裝置，不是盲選 nvidia-smi 第一列；初始化前或讀取失敗顯示未記錄 |
| CPU 型號、實體／邏輯 CPU 數、程序 affinity 可用 CPU 數 | 本次測試環境可見資訊 |
| RAM 總容量 | 執行測試的 OS／Docker VM 可見容量；不是保證等於實體主機安裝容量，也不等於容器 cgroup 限額 |
| 時間、OS、Python、PyTorch、Ultralytics 等版本 | 本次執行時讀取；報告只顯示已保存的快照 |
| 各階段 GPU／VRAM／CPU／RAM 使用量 | 原本的資源採樣，現在也自動列在 report.md |

**Windows 建議使用啟動腳本**，額外擷取實體主機 CPU、安裝 RAM、Windows 可見 RAM 與 OS，和 Docker／WSL 的容量分開呈現：

```powershell
.\run.ps1 -Mode run
# 已匯入或建置好 1.3 映像時，可略過建置
.\run.ps1 -Mode run -NoBuild
```

`scripts/Get-HostInfo.ps1` 會在建置與權重準備完成後重新讀取 Windows CIM，將快照僅傳給本次 Docker 命令。缺少、格式錯誤、過期或未來時間的主機快照不會套用；資訊缺失會標明，不能冒用上一台設備的 RAM。採集的 CPU／RAM 規格本身不會改變驗收標準。

直接執行 `docker compose run --rm benchmark run` 仍會自動記錄測試環境 CPU／RAM 與實際 GPU；**沒有 Windows 腳本快照時，主機實體規格顯示「未提供」**。Linux 裸機使用 OS 可見 RAM，虛擬機／Docker Desktop 則是其可見的 RAM；不從容器數值推算實體 RAM。快照中的 RAM 容量與資源驗收保持同樣的 OS／VM 範圍；Windows 實體安裝容量不作為 VM RAM 百分比的分母。

## 預設驗收條件

條件在 `config.json` 修改，會隨報告保存。每一路都必須通過，不能用平均 FPS 掩蓋較慢的攝影機。

| 項目 | 預設值 |
|---|---|
| AI 分析取樣率 | 每路 10 FPS，每張畫面執行六模型 |
| 每路平均完成 FPS | ≥ 9.5（目標的 95%） |
| 每個 5 秒視窗的完成 FPS | ≥ 9（目標的 90%） |
| 六模型完整端到端延遲 p95 | ≤ 500 ms |
| 未完成／被覆蓋／未及解碼的取樣畫面比例 | ≤ 1% |
| 全 GPU VRAM 已用量峰值 | ≤ 90% |
| 主機／Docker Linux VM RAM 已用比例峰值 | ≤ 90% |
| CPU／GPU utilization | 收集但預設不作為失敗條件；CPU 上限可另設 |
| 模型推論尺寸 | `imgsz=640`；依 Ultralytics 預設保持影像比例縮放 |
| 精度／批次 | FP16、每批 1 張；可調整，但不同批次的結果不能直接比較 |

1% 丟幀限制通常比平均 FPS 的 95% 限制更嚴格。GPU utilization 達到 100% 本身不代表不通過；重點是能否滿足每路需求與延遲。

調整例子：

```sh
# 僅做 10 秒單路檢查（smoke 不使用 test_streams 清單）
docker compose run --rm benchmark smoke --duration 10

# 先在 config.json 設 test_streams=null，才會使用自動搜尋上限
docker compose run --rm benchmark run --max-streams 32

# 加速試跑：10 秒搜尋 + 一次 30 秒確認（不等同長時間穩定性驗證）
docker compose run --rm benchmark run --duration 10 --confirm-duration 30 --confirm-repeats 1

# 比較批次設定，例如每批最多 4 路各取一張畫面
docker compose run --rm benchmark run --batch-size 4

# 使用原版高階 predict 流程作對照
docker compose run --rm benchmark smoke --duration 60 --execution predict
```

## 看報告

**先開 `report.md` 看結論；要比較 FPS、GPU、CPU、RAM，開同資料夾的 `summary.csv`。** CSV 可用 Excel 開啟，包含各階段的通過／失敗、完整 FPS、p95 延遲、丟幀、GPU 使用率、VRAM、CPU 與 RAM。

本機目前的結果請區分：

- **最新完整重測（1.3，自動記錄設備）**：`results/20261022-060211-722ac4/report.md`，1 路通過三輪各 300 秒確認，皆為 10.00 FPS、零丟幀，最差 p95 延遲 61.40 ms；雙路丟幀率 1.33%／1.50% 未通過，後續路數未測。GPU／CPU／RAM 與 Windows 實體主機快照均自動寫入本次 `environment.json`／`summary.json`，報告不需手動補註。
- **前次完整重測（1.2 清單模式）**：`results/20261022-051829-1cf59e/report.md`，每路 10 FPS、六模型，2 路通過三輪各 300 秒確認；4 路未通過，3／8／16 路未測。已註記本次 GPU／CPU／RAM 與長測資源使用量，啟動前主機資訊另存 `host-context.json`。
- **前次完整容量測試（自動搜尋）**：`results/20261021-083911-f628f2/report.md`，每路 10 FPS、六模型，2 路通過三輪各 300 秒確認，3 路未通過。已補註硬體與環境資訊。
- **後續功能短測**：`results/20261021-091308-1f5627/report.md`，雙路僅量測 10 秒，用來確認新版程式可執行，不能取代完整容量長測。不要只挑時間最新的資料夾當作容量結論。

上述 1.2 版完整容量報告的硬體註記是當時另外採集／補註，不代表舊版已自動採集；未改寫它們的原始 JSON／CSV。1.3 起的新測試才會自動擷取並產生設備區塊；新版的功能短測不取代上述容量長測。

主機 `results/<時間-識別碼>/`：

```text
config.json                  此次完整設定
environment.json             本次自動擷取的 CPU／RAM／實際 GPU、選用 Windows 主機快照、軟體版本及檔案 SHA-256
summary.json                 搜尋結果、是否確認、每路指標與失敗原因
summary.csv                  每個路數／確認回合的指標與設備比較欄位，可用 Excel 開啟
report.md                    可閱讀的設備表、驗收結論與資源使用摘要
01-search-01streams/
  config.json                該階段設定（確認階段量測時間可能較長）
  preflight.json             六模型各自的單張真實影片推論檢查
  result.json                該階段詳細數據／錯誤
  resources.csv              每秒 CPU、RAM、GPU、VRAM、可用時的溫度／功耗
  frames.csv                 每個完成影格的捕捉時刻、完成時刻、六模型耗時
  stage.log                  完整執行訊息與例外
```

`summary.json` 的 `max_stable_streams` 是通過指定確認測試的最高已測路數；清單模式僅涵蓋 `test_streams`，不能當作精確硬體上限。`at_limit=true` 表示到達清單最後一筆或自動搜尋的路數上限。`smoke` 或 `backend=synthetic` 時容量欄為 `null`。報告會明確標示 synthetic 不是 GPU 實測。

`results/` 掛載在主機，不會隨容器刪除；但目前被 `.gitignore` 排除，不會隨 Git push／clone 傳送。若要帶到另一台電腦比較，需另外複製對應結果資料夾。

結束碼：0 = 執行完成且有通過路數（或 synthetic 軟體流程完成）；1 = 單路檢查未通過／實測無通過路數；2 = 設定、檔案、CUDA 或軟體錯誤；130 = 使用者中斷。

## 換到另一台電腦

最簡單是複製整個資料夾（至少 `Dockerfile`、`requirements.txt`、`compose.yaml`、`config.json`、`benchmark/`、`tests/`、`scripts/`、`models/`、`video.mp4`，Windows 另帶 `run.ps1`），在新電腦執行上述命令。程式會讀取新電腦的設備，不要複製舊的主機快照來當新設備規格。

為了讓兩台使用**完全相同的已建置映像**，也可匯出：

```sh
docker save -o multi-stream-ai-benchmark.tar multi-stream-ai-benchmark:1.3
```

將映像 tar、`compose.yaml`、`config.json`、`models/` 和 `video.mp4` 複製至新電腦的同一資料夾（Windows 主機規格採集另帶 `run.ps1` 與 `scripts/`），再執行：

```sh
docker load -i multi-stream-ai-benchmark.tar
docker compose run --rm benchmark smoke --duration 15
docker compose run --rm benchmark run
```

有 `models/yolo11m.pt` 時推論不需要下載權重。映像不內嵌影片或權重，方便更換影片／正式模型，也避免把影片帶入 Docker build context。

## 測試方法與結果範圍

每路使用獨立 OpenCV reader 按原影片時間軸循環播放，依目標 AI FPS 取樣。每路只保留最新一張待處理畫面，避免記憶體或延遲無限增長。取樣率必須小於等於影片 FPS，超過時直接報錯，避免用重複畫面虛增攝影機需求。

一個公平輪詢的 GPU 排程器從各路取畫面，支援將不同路畫面組成批次。六份模型只由該排程器存取。`execution=cuda_graph` 會將六個獨立模型的前向運算錄製為一個 CUDA Graph，每次寫入新的影像資料後完整重播六模型。相同的 letterbox、BGR→RGB、正規化及 CPU→GPU 複製只做一次，六個模型的 NMS 與原始影像座標還原仍各自執行。全部完成後才做一次顯式 CUDA 同步並計入完成影格；動態 NMS 內部必要的同步仍存在。

`execution=predict` 保留原流程：逐一呼叫六個模型的高階 predict，每個模型完成後同步。總 FPS 在兩種模式中都只計每秒完成全部六模型的影格數，完成的模型影格推論總數另列為 `completed_model_frame_evaluations`。`model_invocations` 是邏輯模型批次執行次數；CUDA Graph 一次重播等於六個模型批次執行，使用批次時此欄不等於影格數。

CUDA Graph 以固定的影像形狀、精度與批次大小擷取。程式在正式量測前準備 1 到 `min(batch_size, 路數)` 的所有批次大小；正式量測中不進行擷取。它會檢查六個模型物件與參數儲存區互相獨立，並保留不同批次的 YOLO anchors，避免切換批次後誤用記憶體。增加批次上限也會增加 Graph 緩衝區的 VRAM；較大批次不保證符合每路 FPS／延遲要求。Graph 本身會在每台電腦上重新建立，不需攜帶 GPU 專屬的 TensorRT engine。

Graph 模式目前支援 YOLO detection `.pt` 模型；其他任務可選 `--execution predict`。每個影格的完整六模型耗時仍包含前後處理與解碼排隊。Graph 模式另外在 `result.json`／`summary.json` 的 `performance_breakdown` 記錄共享前處理、整組六模型 CUDA event 時間、後處理與等待、完整批次時間，以及每個模型的實際影格評估數。CPU wall time 與 GPU 時間彼此重疊，不能直接相加。Graph 是整組計時，`model_timing_scope=group_cuda_graph` 時各模型的獨立毫秒欄位為空，避免填入不實的分攤時間。

端到端延遲從**預定取樣時刻**算到六模型完成，包含解碼、排隊、前處理、推論、後處理。測試結束後才完成的影格列為未完成，不延長計時補算。`offered` 根據測試時長與取樣率計算，因此解碼太慢也會被算進失敗。延遲百分位只計已完成影格，必須與丟幀率一起解讀。

每個路數使用獨立程序載入六模型，先做預檢，再用相同路數暖機，最後開始正式量測。模型載入、下載、暖機不計入 FPS。VRAM 不足或推論超時會判定該路數失敗，程序結束後釋放 GPU；清單模式停止增加路數，自動搜尋模式才會補測較低路數。檔案損壞、CUDA 不可用等基礎問題則立即停止，保留診斷紀錄。

CPU／RAM 反映 Python 所在的 Linux 主機或 Docker VM，不一定等於 Windows 工作管理員數值；另收集 benchmark 程序 RSS。GPU／VRAM 是整張裝置，包括其他程式使用量。NVML 在部分 WSL 環境不提供 GPU utilization 時保留 `null` 與警告，VRAM 嘗試以 CUDA 記憶體查詢補足；必要驗收資料缺失不會當成 0 或通過。資源峰值來自每秒採樣，極短暫尖峰可能未被採到。

搜尋假設增加路數通常會使效能下降；長測確認用於篩掉短期偶然通過。這份結果表示**此影片、此六模型、取樣率、排程器、精度及測試條件下的系統容量**。它含軟體解碼與檔案 I/O 成本，不含 RTSP 網路、錄影寫入、NVDEC 解碼、追蹤或正式六種不同模型的額外工作。不同 GPU 比較時，使用同一份影片、權重、映像、設定，並保持電源模式與背景負載一致。

正式模型可在 `weights` 列表指定六個不同的本地 `.pt` 路徑，放進 `models/`。六份相同 YOLO11-M 僅模擬六模型計算量；實際模型工作量變更後須重新測試。

## 本地開發與測試

測試可在有 psutil、NumPy、OpenCV 的 Python 3.9+ 環境執行；Docker 使用固定的 PyTorch/CUDA 與套件版本。

```sh
python -m unittest discover -s tests -v
python -m benchmark smoke --backend synthetic --streams 2 --duration 1 --warmup 0
```

GPU 環境完整測試、真實六模型輸出一致性與瓶頸診斷：

```sh
docker compose run --rm --entrypoint python benchmark -m unittest discover -s tests -v
docker compose run --rm --entrypoint python benchmark -m benchmark.validate_optimization
docker compose run --rm --entrypoint python benchmark -m benchmark.diagnose --iterations 20
```

沒有 CUDA 或本地 PyTorch 太舊時，三項 CUDA Graph 硬體測試會明確跳過；在實際 GPU 容器內需確認它們都有執行。`validate_optimization` 會比較影片五個時間點、batch 1／2 切換與返回舊畫面時的六份最終偵測框，逐值完全相同才通過。診斷結果保存在 `results/diagnosis/`，不能當成已通過長時間驗收的攝影機路數。

synthetic 模式不載入影片、模型或 CUDA，只驗證排程、報告和搜尋流程；不會產生可用作 GPU 容量結論的 `max_stable_streams`。

參考：[Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/)、[YOLO predict](https://docs.ultralytics.com/modes/predict/)、[PyTorch CUDA Graphs](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)、[Docker Desktop GPU](https://docs.docker.com/desktop/features/gpu/)。
