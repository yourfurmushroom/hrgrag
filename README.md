# Portable Runner

這個資料夾是可攜版執行包，只保留跑 benchmark / grammar 所需的最小程式碼，不包含原始資料集。

## 內容

- `LLM_inference_benchmark/`: benchmark 主程式與 agent 模組
- `hrg_grammar/`: grammar 抽取程式
- `requirements.txt`: Python 套件
- `setup_env.sh`: 建立 `.venv`、安裝依賴、下載 NLTK `punkt`
- `run_pipeline.sh`: 一鍵先抽 grammar，再跑 benchmark
- `download_datasets.py` / `download_datasets.sh`: 自動下載資料集
- `generate_configs.py`: 根據已下載資料自動產生 `configs/config.*.env`
- `bootstrap_all.sh`: 一鍵完成環境安裝、資料下載、config 生成
- `resolve_kb.py`: 自動尋找已知或常見位置中的 KB
- `auto_benchmark.sh`: 單腳本完成下載、KB 自動解析、benchmark 執行
- `Dockerfile` / `docker-compose.yml`: 容器化一鍵執行
- `config.env.example`: 目標機器可直接改成 `config.env`

## 快速開始

1. 複製這整個資料夾到新機器。
2. 把 `config.env.example` 改成 `config.env`。
3. 放入你自己的 KB 與資料集檔。
4. 安裝環境：

```bash
bash setup_env.sh
```

如果你要直接用 Docker：

```bash
docker compose up --build
```

現在預設會依序跑目前可直接閉環的資料集：

- `metaqa`
- `wikimovies`
- `mlpq`
- `kqapro`

若要只跑單一資料集：

```bash
DATASET=wikimovies docker compose up --build
DATASET=mlpq docker compose up --build
DATASET=kqapro docker compose up --build
```

如果要自動依序跑全部：

```bash
DATASET=all docker compose up --build
```

如果只想跑一組自訂清單：

```bash
DATASETS="metaqa wikimovies mlpq kqapro" docker compose up --build
```

如果機器沒有 Docker，直接用本機一鍵入口：

```bash
bash run_local_all.sh
```

這會自動做環境安裝、資料集下載、config 生成，然後依序跑：

- `metaqa`
- `wikimovies`
- `mlpq`
- `kqapro`

若只想跑指定資料集：

```bash
bash run_local_all.sh metaqa wikimovies
```

容器會把資料與產出保留在這些 bind mounts：

- `./Datasets`
- `./artifacts`
- `./configs`
- `./KBs`
- `./.cache`

5. 全自動初始化：

```bash
bash bootstrap_all.sh
```

這會自動做三件事：

- 建立 `.venv` 並安裝依賴
- 下載預設資料集
- 產生 `configs/config.*.env` 與 `configs/run_*.sh`

6. 或者分步執行下載資料集：

```bash
bash download_datasets.sh
```

也可以只抓指定資料集：

```bash
bash download_datasets.sh --datasets metaqa wikimovies mlpq
```

7. 一鍵執行：

```bash
bash run_pipeline.sh
```

若已跑過 `bootstrap_all.sh`，也可以直接用自動生成的 launcher，例如：

```bash
bash configs/run_metaqa.sh
bash configs/run_wqsp.sh
```

如果你要單腳本一路跑到底，直接：

```bash
bash auto_benchmark.sh metaqa
bash auto_benchmark.sh wikimovies
bash auto_benchmark.sh mlpq
bash auto_benchmark.sh kqapro
```

下載完後，`WQSP`、`CWQ`、`KQA Pro`、`Mintaka` 現在也能直接設：

```bash
DATASET=wqsp
SPLIT=test
KB_PATH=/path/to/your/kb.txt
```

`run_pipeline.sh` 會預設讀：

- `Datasets/WQSP/normalized/test.jsonl`
- `Datasets/CWQ/normalized/test.jsonl`
- `Datasets/KQAPro/normalized/test.jsonl`
- `Datasets/Mintaka/normalized/test.jsonl`

## 目前 downloader 會處理的資料集

- `MetaQA`: 走官方 GitHub README 指到的 Google Drive folder，目標是讓 `MetaQA (Vanilla Test)` 可用。
- `WQSP`: 預設抓 Hugging Face mirror `ml1996/webqsp`，會輸出 `raw/` 與 `normalized/`。
- `CWQ`: 預設抓 Hugging Face mirror `drt/complex_web_questions`。
- `KQA Pro`: 預設抓 Hugging Face mirror `soongfs/kqa_pro`。
- `Mintaka`: 預設抓 `AmazonScience/mintaka`。
- `WikiMovies`: 下載 `movieqa.tar.gz`，保留 `movieqa/` 結構，可對應 `wiki_entities train`。
- `MLPQ`: 下載官方 GitHub repo zip，保留 `datasets/`、`resources/`、`baselines/`。

## Canonical Layout

下載完成後，`portable_runner/Datasets/` 的標準結構應該是：

- `Datasets/MetaQA/`
  直接包含 `1-hop/`、`2-hop/`、`3-hop/`、`kb.txt`
- `Datasets/WikiMovies/`
  直接包含 `movieqa/`
- `Datasets/MLPQ/`
  直接包含 `datasets/`、`resources/`、`baselines/`
- `Datasets/WQSP/`
  直接包含 `raw/`、`normalized/`
- `Datasets/CWQ/`
  直接包含 `raw/`、`normalized/`
- `Datasets/KQAPro/`
  直接包含 `raw/`、`normalized/`，若有完整 snapshot 則放在 `hf_snapshot/`
- `Datasets/Mintaka/`
  直接包含 `raw/`、`normalized/`

下載器現在會嘗試自動把「外面多包一層資料夾」的情況 flatten 回這個 canonical layout。

所有非資料集產物則統一放在：

- `artifacts/<run_tag>/grammar/`
- `artifacts/<run_tag>/results/`
- `artifacts/<run_tag>/dumps/`

其中：

- grammar JSON/TXT 放在 `artifacts/<run_tag>/grammar/`
- benchmark report 與 wide CSV 放在 `artifacts/<run_tag>/results/`
- question dump、shared retrieval cache、prepared pickle 放在 `artifacts/<run_tag>/dumps/`

## 目前已知限制

- `WQSP`、`CWQ`、`KQA Pro`、`Mintaka` 這四個目前是用我已知可穩定存取的 Hugging Face mirror，不是我直接驗到的原始官方壓縮包。
- `run_pipeline.sh` 現在已可直接讀 `WQSP`、`CWQ`、`KQA Pro`、`Mintaka` 的 `normalized/*.jsonl`；但它們仍然需要你自己提供對應的 `KB_PATH`，否則 benchmark 不會有可檢索的知識圖譜。
- `MetaQA` 來源是 Google Drive folder；若官方調整分享設定，`gdown` 之後可能需要改。
- `configs/config.*.env` 會盡量自動填可推斷的路徑；對 `WQSP`、`CWQ`、`KQA Pro`、`Mintaka`，`KB_PATH` 仍需要你自己換成實際可用的 triples / KG 檔。
- `auto_benchmark.sh` 會先嘗試自動解析 KB。`MetaQA`、`WikiMovies`、`MLPQ` 以及通常的 `KQA Pro` 可以直接自動命中；`WQSP`、`CWQ`、`Mintaka` 若原始資料沒有附 KB，腳本只會掃描 `portable_runner/KBs/`、`portable_runner/Datasets/` 等常見位置，找不到就會停止並提示你補檔。

## Custom Dataset 格式

`DATASET=custom` 時支援兩種格式：

- `jsonl`: 每行一筆，例如 `{"question":"Who directed [Inception]?","answers":["Christopher Nolan"],"hop":1}`
- `tsv`: `question<TAB>answer1|answer2<TAB>hop`

`hop` 可省略，缺省時使用 `CUSTOM_HOP`。

## 備註

- 這份 portable 版新增了 `--dataset custom` 與 `--model-filter`，方便換資料集和只跑單一模型。
- `torch` 安裝方式依新機 CUDA 環境可能不同；若需要指定版本，可先設 `TORCH_INSTALL_CMD` 再執行 `setup_env.sh`。
