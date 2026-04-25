# XP-Gacha / 地下金库

一个基于 `Streamlit` 的本地（正经）漫画库存管理、检索与推荐系统。

它把「数据抓取 / CSV 清洗 / MySQL 入库 / 标题分词 / 标签加权 / 向量语义检索 / LLM 问答」串成了一条完整流程，适合做个人向的漫画资料库、XP 标签筛选器和本地图库浏览器。

当前实现明显偏向 Windows 本地环境使用：

- 支持直接打开本地文件夹（`os.startfile`）
- 默认配置里使用了 Windows 绝对路径
- 本地模型与图库目录默认按本地磁盘组织

## 功能概览

- 基于 `MySQL` 读取漫画元数据并缓存预处理结果
- 按标签、作者、标题关键词进行动态推荐评分
- 支持屏蔽标签、权重调节、分数阈值筛选
- 支持普通关键词搜索
- 支持本地向量模型的自然语言语义检索
- 支持展示封面缩略图、来源链接和本地目录
- 支持在界面中直接打开本地漫画文件夹
- 支持将当前筛选结果注入给 LLM 做问答
- 附带在线抓取、CSV 入库、向量库构建等数据脚本

## 项目结构

```text
XP-Gacha/
├─ app.py                                  # Streamlit 主入口
├─ config_empty.py                         # 配置模板
├─ config.py                               # 本地实际配置（默认未纳入版本控制）
├─ data_pipeline.py                        # 数据读取、缓存、标签/标题处理、动态评分
├─ utils_core.py                           # 本地目录匹配、封面缩略图与 Base64 缓存
├─ utils_nlp.py                            # 标题分词、语义检索模型加载
├─ utils_chat.py                           # LLM 对话与流式输出
├─ .streamlit/
│  └─ secrets.toml                         # MySQL 密钥配置
├─ dictionaries/                           # 停用词、语义映射等字典资源
│  ├─ SEMANTIC_MAP.json
│  ├─ STOP_TAGS.txt
│  ├─ TITLE_SEMANTIC_MAP.json
│  └─ TITLE_STOP_WORDS.txt
├─ data/                                   # 原始数据、整理后的 CSV 与 SQL 备份
│  ├─ gallery_info/                        # 已补全文件夹名的标准化 CSV
│  ├─ gallery_info_no_name/                # 原始抓取 CSV
│  └─ local_data/                          # 本地书签/链接列表输入
├─ data_get/                               # 在线抓取、本地链接补全等脚本
│  ├─ get_information_online.py
│  ├─ get_information_online_fix.py
│  └─ local/
│     ├─ get_images_local.py
│     ├─ get_information_local.py
│     └─ output/                           # 本地抓取输出目录
├─ data_processing/                        # CSV 入库、向量库构建、预处理脚本
│  ├─ add_csv_to_mysql.py
│  ├─ addname.py
│  ├─ all_csv_to_mysql.py
│  ├─ b64_pre_encode.py
│  ├─ build_vector_db.py
│  ├─ map_add_name.py
│  ├─ tag_set.py
│  └─ title_cut_set.py
├─ manga_vectors/                          # 向量库输出目录
├─ onlineimgtmp/                           # 在线封面缩略图缓存
├─ localimgtmp/                            # 本地封面缩略图缓存
├─ b64_cache/                              # Base64 封面缓存
├─ b64_tmp/                                # Base64 处理中间目录
├─ datacache/                              # DataFrame 预处理缓存
├─ paraphrase-multilingual-MiniLM-L12-v2/  # 本地 embedding 模型目录（可替换）
└─ Qwen3-Embedding-0.6B/                   # 本地 embedding 模型目录（默认）
```

## 核心流程

1. 抓取或整理漫画元数据，落成 CSV。
2. 将 CSV 写入 MySQL 表 `gallery_info`。
3. 使用本地 embedding 模型为数据库内容构建向量库。
4. 启动 Streamlit 页面。
5. 页面启动后会读取数据库，做标签清洗、标题分词和本地目录匹配。
6. 用户在侧边栏调节权重、屏蔽标签、关键词搜索或语义检索。
7. 前端展示筛选后的库存，可继续进行语义检索，并可调用本地或线上 LLM 实时RAG问答。

## 运行环境

建议环境：

- Python `3.10+`
- Windows
- 可用的 MySQL 实例
- 下载到本地的 embedding 预训练模型
- 如需本地聊天：已启动的 `LM Studio` 兼容接口

安装依赖包：

```bash
pip install -r requirements.txt
或
pip install streamlit pandas sqlalchemy pymysql pillow janome sentence-transformers torch requests curl-cffi beautifulsoup4 tomli
```

## 配置

### 1. 创建 `config.py`

先复制模板：

```bash
copy config_empty.py config.py
```

然后按本机环境修改 `config.py` 中的关键配置：

- `BASE_DIR`：本地漫画根目录
- `LOCAL_MODEL_PATH`：本地 embedding 模型目录
- `VECTOR_FILE`：向量文件输出位置
- `LM_STUDIO_API_BASE` / `LM_STUDIO_MODEL`：本地模型服务
- `ONLINE_API_BASE` / `ONLINE_API_KEY` / `ONLINE_MODEL`：线上模型服务
- `INITIAL_TAG_WEIGHTS`：默认标签权重
- `MAX_DISPLAY`：单页最大显示条目数

`config_empty.py` 里已经包含了完整字段，可以直接作为模板使用。

### 2. 配置数据库密钥

在 `.streamlit/secrets.toml` 中提供 MySQL 连接信息：

```toml
[mysql]
user = "root"
password = "your_password"
host = "127.0.0.1"
port = 3306
database = "gallery_info"
```

应用和数据脚本都会从这里读取数据库配置。

### 3. 准备字典与模型资源

项目默认会读取：

- `dictionaries/STOP_TAGS.txt`
- `dictionaries/SEMANTIC_MAP.json`
- `dictionaries/TITLE_STOP_WORDS.txt`
- `dictionaries/TITLE_SEMANTIC_MAP.json`
- `Qwen3-Embedding-0.6B/` 或你在 `config.py` 中指定的本地模型目录

## 启动项目

```bash
streamlit run app.py
```

启动后页面提供以下能力：

- 算法依据库存偏向自动评分
- 标签权重调节
- 作者偏好加权
- 标题关键词加权
- 屏蔽标签
- 关键词搜索
- AI 语义向量检索
- AI 语义向量评分
- 多表头信息全局排序与当前页排序
- 当前结果集RAG-LLM问答
- 一键跳转网页链接打开
- 检索本地漫画目录并打开
- 总库存偏好图表统计

## 数据准备与维护

### 在线抓取元数据

脚本会循环抓取指定页数范围，抓取信息写入*.csv，并下载缩略图到 `onlineimgtmp/`：

例：循环抓取线上1~100页的漫画信息，抓取tag分类需调整代码

```bash
python data_get/get_information_online.py 100
```

### 补充抓取元数据

脚本会重新抓取错误报告内的页码：

```bash
python data_get/get_information_online_fix.py
```

### 从本地链接列表抓取元数据

如果你已经有本地整理过的链接文本（浏览器导出书签转txt格式），也可以使用：

```bash
python data_get/local/get_information_local.py
```

### 标准化 CSV 数据（添加从本地链接列表抓取的文件夹名）

脚本会加工*.csv为*_full.csv

```bash
python data_processing/addname.py
```

### 全量 CSV 导入 MySQL

会读取 `data/gallery_info/*.csv`，合并后覆盖写入 `gallery_info`：

```bash
python data_processing/all_csv_to_mysql.py
```

### 增量同步 CSV 到 MySQL

适合持续追加数据后同步：

```bash
python data_processing/add_csv_to_mysql.py
```

### 构建向量库

读取 MySQL 中的 `gallery_info`，生成 `VECTOR_FILE` 指向的向量文件：

```bash
python data_processing/build_vector_db.py
```

### 批量下载本地链接列表漫画图片

```bash
python data_get/local/get_images_local.py
```

## 界面中的评分逻辑

项目的推荐分大致由三部分组成：

- 标签分：基于标签频次、标签权重和全局倍率
- 作者分：基于作者出现频次和作者偏好倍率
- 标题分：基于标题分词后的高频词与词权重

最后会按 `推荐评分` 和 `上传日期` 进行排序展示。

这套逻辑的目标不是“客观评分”，而是把你的历史收藏偏好显式化，做成一个可调的个人 XP 排序器。

## 缓存说明

为了提升启动速度，项目会生成几类缓存：

- `datacache/`：预处理后的 DataFrame 缓存
- `localimgtmp/`：本地封面缩略图缓存
- `onlineimgtmp/`：在线抓取到的缩略图
- `b64_cache/`：封面 Base64 缓存

当数据库内容或字典文件变化后，应用会自动根据哈希重新生成预处理缓存。

## 注意事项

- 当前实现对 Windows 更友好，尤其是“打开本地文件夹”功能。
- `BASE_DIR`、模型路径等默认值是本机路径，换机器必须修改。
- 语义检索依赖本地 embedding 模型和提前构建好的向量文件。
- `app.py` 启动时会连接数据库；如果密钥或表不存在，页面会直接报错停止。
- 若使用在线抓取脚本，默认代理地址写死为 `127.0.0.1:7890`，需要按实际网络环境调整。

## 适合谁用

如果你想要一个个人化、可解释、可调权重、同时支持本地浏览和 AI 检索的（正经）漫画库存系统，这个项目非常适合你的需求:P。
