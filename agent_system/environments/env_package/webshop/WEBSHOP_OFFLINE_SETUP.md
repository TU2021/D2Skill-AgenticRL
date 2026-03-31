# WebShop 环境离线配置指南

当本机无法访问外网时，需要先在**能联网的电脑**上下载好数据和依赖，再拷贝到当前机器的指定目录。按下面步骤操作即可。

---

## 一、需要下载的数据文件

在**能联网的电脑**上，从 Google Drive 下载以下文件（可用浏览器或 `gdown` 等工具）。

### 1. 最小配置（small，推荐先做）

用于 `use_small: true` 或 1000 商品规模，本仓库代码默认使用该配置。

| 文件名（下载后请重命名） | Google Drive 链接 | 说明 |
|--------------------------|-------------------|------|
| `items_shuffle_1000.json` | https://drive.google.com/uc?id=**1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib** | 1000 条商品抓取信息 |
| `items_ins_v2_1000.json`  | https://drive.google.com/uc?id=**1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu** | 1000 条商品属性/指令 |
| `items_human_ins.json`    | https://drive.google.com/uc?id=**14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O** | 人类指令数据（必选） |

### 2. 全量配置（all，可选）

若需要完整 1.18M 商品与 12k+ 指令，额外下载：

| 文件名（下载后请重命名） | Google Drive 链接 |
|--------------------------|-------------------|
| `items_shuffle.json` | https://drive.google.com/uc?id=**1A2whVgOO0euk5O13n2iYDM0bQRkkRduB** |
| `items_ins_v2.json`  | https://drive.google.com/uc?id=**1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi** |

### 3. 可选数据（按需）

- **ResNet 图像特征**（仅做需要图像特征的模型时用）：  
  https://drive.google.com/drive/folders/1jglJDqNV2ryrlZzrS0yOEk-aRAcLAhNw  
  下载后把解压出的文件放到下面的 `data/` 目录。
- **人类示范轨迹**（可选）：  
  https://drive.google.com/file/d/1GWC8UlUzfT9PRTRxgYOwuKSJp4hyV1dp/view

---

## 二、拷贝到本机的目录

将上面下载并重命名好的文件放到：

```text
SkillRL/agent_system/environments/env_package/webshop/webshop/data/
```

即项目内 webshop 的 **data** 目录，绝对路径示例：

```text
/data/home/zdhs0006/SkillRL/agent_system/environments/env_package/webshop/webshop/data/
```

目录结构应为：

```text
webshop/webshop/data/
├── items_shuffle_1000.json   # 最小配置必选
├── items_ins_v2_1000.json    # 最小配置必选
├── items_human_ins.json      # 必选
├── items_shuffle.json        # 全量时可选
└── items_ins_v2.json        # 全量时可选
```

**注意**：文件名必须与上表一致（尤其是 `items_shuffle_1000.json`、`items_ins_v2_1000.json`），否则代码会找不到文件。

---

## 三、本机还需完成的步骤（无需外网）

数据就位后，在**当前机器**上在 webshop 环境中执行以下步骤（可不联网进行，只要依赖已装好）。

### 1. 安装运行依赖（若尚未安装）

需提前装好（可用离线包或内网源）：

- Python 3.8+
- **Java 11**（建 Lucene 索引用）：`conda install -c conda-forge openjdk=11` 或系统安装
- 项目依赖：在 `webshop/webshop/` 下执行  
  `pip install -r requirements.txt`  
  （若本机完全无网，需在能联网机器打包 wheel 再拷贝安装）
- **faiss-cpu**：`conda install -c conda-forge faiss-cpu`
- **spaCy 英文模型**（必须）：  
  - `en_core_web_sm`（必选）  
  - `en_core_web_lg`（可选）  
  若本机不能联网，可在有网机器下载 wheel 再拷贝安装，例如：  
  https://github.com/explosion/spacy-models/releases（选对应 Python 版本）

### 2. 生成检索资源并建索引

在 **search_engine** 目录下执行（会读取 `data/` 里的 `items_*`，并生成 Lucene 索引）：

```bash
cd /data/home/zdhs0006/SkillRL/agent_system/environments/env_package/webshop/webshop/search_engine

# 根据 data 中的 items 生成 resources_100 / resources / resources_1k / resources_100k
python convert_product_file_format.py

# 建 Lucene 索引（需要已安装 pyserini 和 Java 11）
./run_indexing.sh
```

若当前只用 1000 商品（small），`utils.py` 里 `DEFAULT_FILE_PATH` / `DEFAULT_ATTR_PATH` 指向的已是 `items_shuffle_1000.json` 和 `items_ins_v2_1000.json`，无需改配置；若要用全量，再按 README 修改 `web_agent_site/utils.py` 中的 `DEFAULT_ATTR_PATH` / `DEFAULT_FILE_PATH` 为 `items_ins_v2.json` 和 `items_shuffle.json`，然后重新执行上面两步。

### 3. 验证环境是否可用

**方式一：用项目自带的验证脚本（推荐）**

在**已安装 webshop 依赖**的环境（即安装过 `requirements.txt`、gym、spaCy 等）下，在 **SkillRL 项目根目录**执行：

```bash
# 激活你的 conda/venv 后
cd /path/to/SkillRL
PYTHONPATH=. python agent_system/environments/env_package/webshop/verify_webshop_env.py
```

若输出以 `========== 验证通过：skill-webshop 环境可正常使用 ==========` 结尾，说明环境配置正确。

**方式二：用 webshop 自带的文本环境示例**

进入 webshop 子目录，用其自带的脚本跑一小段（会加载商品、做一次检索并执行一步）：

```bash
cd agent_system/environments/env_package/webshop/webshop
python run_web_agent_text_env.py
```

若能看到 “Products loaded.”、“Keys Cleaned.”、“Attributes Loaded.” 以及若干步 “Taking action ... -> Reward = ...” 则说明环境与检索索引正常。

若报错缺少某文件，根据报错路径检查 `data/` 与 `search_engine/indexes*` 是否齐全。

### 4. 初始化时间与日志提示

- **单进程（如验证脚本）**：small（约 1000 商品）通常 **30 秒～2 分钟**（读 JSON、建索引、加载 goals）。全量数据会明显更久。
- **训练时（多 Ray worker）**：每个 worker 各自做一遍上述加载；总时间取决于 worker 数量与机器。控制台会先看到 `[WebShop] Initializing WebShop envs ...`，然后各 worker 输出：
  - `Products loaded.` → `Keys cleaned.` → `Attributes loaded.` → **tqdm 进度条**（商品逐条处理）→ `Loaded N goals.`
- 之后会打印 `[WebShop] Waiting Xs for all workers to finish init...` 和 `[WebShop] Envs ready.`，表示环境就绪。

---

## 四、小结

| 步骤 | 在哪里做 | 做什么 |
|------|----------|--------|
| 1 | 能联网的电脑 | 从 Google Drive 下载 3 个必选 JSON（small）或 5 个（all），并按表重命名 |
| 2 | 本机 | 把上述文件拷贝到 `.../webshop/webshop/data/` |
| 3 | 本机 | 安装 Python、Java 11、pip/conda 依赖、faiss、spaCy 模型（可离线包） |
| 4 | 本机 | 在 `webshop/webshop/search_engine/` 下执行 `convert_product_file_format.py` 和 `run_indexing.sh` |

完成以上步骤后，skill-webshop 环境即可在本机无外网的情况下使用。
