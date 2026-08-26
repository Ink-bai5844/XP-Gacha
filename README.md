# XP-Gacha / 地下金库

XP-Gacha 是一个面向个人漫画馆藏的检索、评分、推荐与数据维护工具。当前主程序已经从 Streamlit 重构为 React 单页应用 + FastAPI API + MySQL；旧 Streamlit 入口仍保留用于兼容和对照，但不再是推荐入口。

**当前版本：`v0.2.5`**

| 项目 | 当前实现 |
| --- | --- |
| Web 前端 | React 18、TypeScript、Vite |
| 应用服务 | FastAPI、Uvicorn |
| 数据库 | MySQL 8.4、`utf8mb4`；全文索引优先使用 `ngram`，失败时回退普通 `FULLTEXT` |
| 推荐计算 | Pandas、NumPy、SciPy 稀疏矩阵 |
| AI 能力 | Qwen3 Embedding、CLIP、LM Studio / OpenAI 兼容 API |
| 部署方式 | Windows 便携包、Docker Compose、源码运行 |
| 默认主访问地址 | `http://127.0.0.1:8000` |
| 默认 API 文档 | `http://127.0.0.1:8000/api/docs` |

> [!IMPORTANT]
> 这是没有登录鉴权的私人本地工具。不要把 Web/API 直接暴露到公网；如需远程访问，请在前面增加访问控制、HTTPS 和可信反向代理。

## 选择运行方式

| 使用场景 | 推荐方式 | 需要预装 |
| --- | --- | --- |
| Windows 用户直接使用 | Windows 11 x64 便携版 | 无 |
| 本机或服务器统一部署 | Docker Compose | Docker Desktop / Docker Engine |
| 修改前后端和调试 | 源码开发 | Python、Node.js、pnpm、MySQL |
| 对照旧界面 | Legacy Streamlit | 源码开发环境 |

## Windows 一键便携版

便携版是普通用户的首选。它内置 CPython 3.13.15、MySQL 8.4.11 noinstall、CPU 版 PyTorch、Python 依赖、React 构建产物和 app-local VC++ 运行库，不要求安装 Python、Node.js、MySQL 或 Docker，也不会注册 Windows 服务或修改全局 `PATH`。

当前构建与启动验证以 Windows 11 x64 为正式支持目标；Windows 10 未纳入本项目的正式验证范围。

### 第一次启动

1. 完整解压 `XP-Gacha-v0.2.5-portable-win64.zip`，不要在压缩软件预览窗口中直接运行。
2. 把目录放到普通本地可写位置。不要放进 `Program Files`、只读目录、网络盘或同步盘。
3. 双击 `Start XP-Gacha.cmd`。
4. 首次启动会在发行包根目录创建并列数据目录、初始化包内 MySQL、随机生成数据库凭据、启动应用并在健康检查通过后打开浏览器。
5. 空数据库的库存首页会显示完整导入引导：打开 `Ink-bai/XP-Gacha-datasets` 下载 `input_data.zip`，保持 ZIP 原样，再跳转“附录 → 一键导入词典 / 数据”上传。

启动窗口需要保持开启。正常停止请在窗口中按 `Ctrl+C`，或双击 `Stop XP-Gacha.cmd`。不要在 MySQL 写入期间直接结束进程或复制数据库目录。

网页 `8000` 和 MySQL `3307` 都只是首选端口；若被占用，启动器会自动寻找可用端口。请以启动窗口打印的实际网址为准。

### 便携包内常用入口

| 文件 | 用途 |
| --- | --- |
| `Start XP-Gacha.cmd` | 启动当前这一份包内 MySQL、API 和 Web |
| `Stop XP-Gacha.cmd` | 安全停止当前包实例 |
| `Check XP-Gacha.cmd` | 检查必需文件和 Python 模块能否加载 |
| `Open XP-Gacha Folder.cmd` | 打开发行包根目录及其并列数据目录 |
| `portable-settings.env` | 可选端口、漫画目录、LLM/API 和在线封面设置 |
| `BUILD-INFO.json` | 版本、源码状态、运行时版本和构建验证结果 |
| `requirements-lock.txt` | 实际打入发行版的 Python 包版本 |
| `SHA256SUMS.txt` | 发行目录内逐文件 SHA-256 |

`Check XP-Gacha.cmd` 不会启动 MySQL，也不验证网络或文件哈希；它通过不代表完整启动一定成功。ZIP 同目录的 `.sha256` 用于核对下载包整体，包内 `SHA256SUMS.txt` 用于核对逐文件完整性。逐文件清单针对刚解压、尚未使用的发行包；启动后生成数据库、缓存或修改根目录词典，相关文件变化是正常现象。

### 便携版设置

`portable-settings.env` 只读取下列白名单键，未知键会被忽略：

| 配置 | 默认值 | 说明 |
| --- | --- | --- |
| `XP_GACHA_PORT` | `8000` | 首选网页端口 |
| `MYSQL_PORT` | `3307` | 首选包内 MySQL 端口 |
| `XP_GACHA_LIBRARY_PATH` | `library` | 漫画目录；相对路径以发行包根目录为基准，也可用绝对路径 |
| `XP_GACHA_IMPORT_MAX_MB` | `1024` | ZIP/CSV 上传上限，单位 MB |
| `MAX_DISPLAY` | `500` | 每页最多显示的库存条数 |
| `LM_STUDIO_API_BASE` | `http://127.0.0.1:1234/v1` | 本地 OpenAI 兼容接口 |
| `LM_STUDIO_API_KEY` | 空 | 本地接口密钥；LM Studio 不要求鉴权时可留空 |
| `LM_STUDIO_MODEL` | `local-model` | 本地模型名 |
| `ONLINE_API_BASE` | 空 | 在线 OpenAI 兼容 API 地址 |
| `ONLINE_API_KEY` | 空 | 在线 API 密钥 |
| `ONLINE_MODEL` | `deepseek-v4-flash` | 在线模型名 |
| `ONLINE_COVER_PROXY` | 空 | 在线封面与 NH 采集代理；空值表示直连，本机代理可填 `http://127.0.0.1:7890` |
| `ONLINE_COVER_FETCH_ENABLED` | `1` | 是否允许在线补抓封面 |

助手页可以直接填写本地或线上 API 的 URL、模型名和 Key；保存时会更新包根目录的 `portable-settings.env`，下一次对话立即使用新配置，无需重启。页面和配置读取接口只会显示 Key 是否已配置，不会把已保存的 Key 明文回传；输入框留空会保留原 Key，只有点击“清除 Key”并保存才会删除。

`portable-settings.env` 可能包含 API 密钥，不要公开；升级时需要单独逐项合并。

`config/portable.json` 保存随机生成的 MySQL 账户和密码。它必须与根目录 `mysql` 成套备份和迁移；不要单独删除、重建或分享该文件。

## Docker Compose

Docker 方式会启动 MySQL 8.4 与应用容器，并把项目的数据、字典、缓存、模型和漫画目录挂载进去。前端在 Node 22 构建阶段编译，运行镜像使用 Python 3.11 和 CPU 版 PyTorch。

### 启动与停止

Windows PowerShell：

```powershell
.\scripts\start.ps1
```

Linux / macOS shell：

```bash
sh ./scripts/start.sh
```

首次启动脚本会从 `.env.example` 复制出 `.env`。默认访问地址为：

- Web：`http://127.0.0.1:8000`
- 健康检查：`http://127.0.0.1:8000/api/health`
- API 文档：`http://127.0.0.1:8000/api/docs`

启动脚本会等到 Web 服务能够正常返回页面后才打印 `started`。库存较大且预处理缓存失效时，全量读取、标签解析和评分缓存会在后台继续构建；网页与健康监控不会再被这一步阻塞，库存目录会在预热完成后自动显示数据。

如果修改了 `.env` 中的 `XP_GACHA_PORT`，请把上述网址的 `8000` 替换为实际端口。

停止服务：

```powershell
.\scripts\stop.ps1
```

或：

```bash
docker compose down
```

普通 `docker compose down` 不删除 MySQL 命名卷。不要使用 `docker compose down -v`，除非确实要永久删除数据库。

### Docker 环境配置

编辑项目根目录 `.env`：

| 变量 | 示例 / 默认值 | 作用 |
| --- | --- | --- |
| `XP_GACHA_PORT` | `8000` | Web 宿主机端口 |
| `MYSQL_EXPOSE_PORT` | `3307` | MySQL 宿主机端口，仅绑定 `127.0.0.1` |
| `MYSQL_DATABASE` | `xp_gacha` | 数据库名 |
| `MYSQL_USER` | `xp_gacha` | 应用账户 |
| `MYSQL_PASSWORD` | `xp_gacha` | 应用账户密码 |
| `MYSQL_ROOT_PASSWORD` | `xp_gacha_root` | MySQL root 密码 |
| `XP_GACHA_LIBRARY_PATH` | `./library` | 宿主机漫画目录，只读挂载到 `/library` |
| `LM_STUDIO_API_BASE` | `http://host.docker.internal:1234/v1` | 容器访问宿主机 LM Studio |
| `LM_STUDIO_API_KEY` | 空 | 本地接口密钥；LM Studio 不要求鉴权时可留空 |
| `LM_STUDIO_MODEL` | `local-model` | 本地模型名 |
| `ONLINE_API_BASE` | 空 | 在线兼容 API 地址 |
| `ONLINE_API_KEY` | 空 | 在线 API 密钥 |
| `ONLINE_MODEL` | `deepseek-v4-flash` | 在线模型名 |
| `ONLINE_COVER_PROXY` | 空 | 在线封面与 NH 采集代理；宿主机代理应填 `http://host.docker.internal:7890`，不能填 `127.0.0.1` |
| `ADMINER_PORT` | `8081` | 可选 Adminer 端口 |

Docker 运行时，助手页保存的 LLM 配置会通过 `/app/.env` 的文件挂载写回宿主项目根目录 `.env`，并在下一次对话请求中生效。无需为助手页保存操作重建或重启容器；如果是在容器外手动编辑 `.env`，仍应重新创建应用容器以重新注入其他 Compose 环境变量。

如果数据库卷已经初始化，修改 `.env` 中的 MySQL 密码不会自动修改现有账户。请在第一次建库前设好密码；已有库应通过数据库命令迁移凭据。

可选启动 Adminer：

```bash
docker compose --profile tools up -d adminer
```

打开 `http://127.0.0.1:8081`，服务器填写 `mysql`，再使用 `.env` 中的数据库名和应用账户。

> [!WARNING]
> Compose 中 Web、MySQL 和 Adminer 的宿主端口默认都只绑定 `127.0.0.1`。如果自行修改端口映射并向局域网或公网开放，必须设置强密码，并增加防火墙、反向代理、HTTPS 和鉴权；助手页的 LLM 配置管理接口仍会拒绝非本机地址和非本机页面发起的请求。

## 从源码运行

### 前置条件

- Python 3.11 或更高版本；
- Node.js 22 与 pnpm 11；
- MySQL 8.x；完整全文检索和导入行为以 MySQL 8.4 为准；
- Windows、Linux 或 macOS 的普通开发环境。

项目已经包含无密钥的 `config.py`，不需要再从 `config_empty.py` 复制。机器相关配置统一使用环境变量。

### 安装、构建和启动

PowerShell 示例：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

corepack enable
pnpm --dir web install --frozen-lockfile
pnpm --dir web build

$env:XP_GACHA_HOST = "127.0.0.1"
$env:DATABASE_URL = "mysql+pymysql://xp_gacha:your_password@127.0.0.1:3306/xp_gacha?charset=utf8mb4"
python launcher.py
```

`launcher.py` 会用 FastAPI 同时提供 `/api/*` 和 `web/dist`，并默认打开浏览器。

开发时可以分两个终端运行：

```powershell
# 终端 1：API，端口 8000
$env:XP_GACHA_HOST = "127.0.0.1"
$env:DATABASE_URL = "mysql+pymysql://xp_gacha:your_password@127.0.0.1:3306/xp_gacha?charset=utf8mb4"
uvicorn server.main:app --host 127.0.0.1 --port 8000 --reload
```

```powershell
# 终端 2：Vite；/api 自动代理到 127.0.0.1:8000
pnpm --dir web dev
```

Python 源码模式会自动读取项目根目录 `.env`，但不会覆盖进程启动前已经存在的同名环境变量；也可以继续在 shell 中设置变量，或由进程管理器注入。助手页保存 LLM 配置时，源码模式会写入这份根目录 `.env`，并同步更新当前进程，因此下一次对话立即生效。

数据库连接优先级为：

1. `DATABASE_URL`；
2. `MYSQL_HOST`、`MYSQL_PORT`、`MYSQL_DATABASE`、`MYSQL_USER`、`MYSQL_PASSWORD`；
3. 旧版 `.streamlit/secrets.toml`，仅兼容兜底；
4. 本机默认 `xp_gacha@127.0.0.1:3306/xp_gacha`。

## 当前页面与功能

| 路由 | 页面 | 主要功能 |
| --- | --- | --- |
| `/` | 库存 | 搜索、评分、筛选、排序、分页、封面、行内详情、可选悬停大图预览、复制当前页 |
| `/detail/:id` | 漫画详情 | 完整元数据、原始与解析标签、打开本地目录、来源跳转 |
| `/chat` | LLM 助手 | 本地/在线兼容 API、连接配置保存、深度思考与回答双流、当前页随机 RAG、Markdown 对话 |
| `/history` | 历史记录 | 刷新、逐条/批量删除、确认清空、重新打开；历史偏好图表统一在“偏好图表”页查看 |
| `/charts` | 偏好图表 | 全局/历史切换，标签、作者、标题词 Top 15 与 Top 150 |
| `/admin` | 附录 | 一键导入、系统状态、26 个白名单数据任务、输出与中止 |
| `/api/docs` | API 文档 | FastAPI 交互式 OpenAPI 文档 |

前端连接不到 API 时会显示 10 条内置虚构样本，用于检查界面布局；这些样本不是实际库存，也不会写入数据库。

### 库存检索

- 关键词覆盖 ID、标题、标题译文、标签、作者和团队。
- MySQL 有全文索引时优先使用 `FULLTEXT`；无索引、无命中或全文查询异常时自动回退 `LIKE`。
- 可选择是否把关键词相关度加入结果列和排序。
- 文本语义检索使用本地 embedding 模型和向量索引。
- 封面相似检索可输入库存 ID，或上传 JPG、JPEG、PNG、WebP、BMP；上传上限 20 MB。
- 标签、作者和标题词选择器无需先搜索即可每批浏览 80 项，也支持关键词搜索、上一批/下一批、已选项移除和清空。
- 每页数量由后端 `config.py` 的 `MAX_DISPLAY` 决定，当前默认 500，不再写死为 5 条。

### 评分与筛选

推荐分由标签、作者、标题词和历史偏好四部分组成：

```text
推荐分 = 标签贡献 + 作者贡献 + 标题词贡献 + 历史偏好贡献
```

标签、作者和标题词的基础分使用全库频次计算：

```text
基础特征分 = 10 × ln(1 + 全库出现次数)
```

多值特征按条目内特征数量的平方根归一化，随后应用单项权重和全局倍率。当前实现还包含两个基线常量：标签贡献额外乘 `0.5`；作者权重向量的默认值是 `5.0`，所以即使作者自定义映射为空，所有已识别作者也会以单项权重 `5.0` 参与计算。历史偏好会按最近打开记录中的标签、标题词和作者计数，并结合该特征在全库中的稀有程度生成附加分。

界面提供：

- 标签、作者、标题、历史偏好四个全局倍率，范围 `0–5`；
- 屏蔽标签；
- 标签、作者、标题词的单项权重，范围 `0–20`；
- 最低推荐评分阈值，范围 `0–1600`。

当前 React Web 启动时四个全局倍率均为 `1`，屏蔽列表和三组自定义映射均为空；这里的“作者映射为空”不改变后端作者权重基线 `5.0`。新选标签/标题词的初始值为 `1`，新选作者的初始值为 `5`，选择作者后才能在界面中改写该作者的数值。`config.py` 中的 `INITIAL_TAG_WEIGHT_NTR` 目前只供旧 Streamlit 使用，不会自动注入 React Web；“纯爱、百合、兽耳”等也没有硬编码默认项，需要在“标签权重配置”中搜索或分批浏览后自行选择。

为避免大库频繁重算：

- 普通筛选请求有约 220 ms 防抖；
- 滑块拖动时只更新预览，松开后提交最终值；
- 数字权重输入约 180 ms 合并更新；
- 后端缓存最近 8 组查询，并把历史文件状态纳入缓存键；
- 评分使用 NumPy/SciPy 向量和稀疏矩阵计算。

### 固定布局与行内详情

- 数据库可用但尚无馆藏时，库存页会隐藏无意义的排序与表格工具，显示从 Hugging Face 下载 `input_data.zip`、跳转一键导入、等待完成并返回库存的四步引导；数据库断连或仅筛选无结果时不会误显示该引导。
- 表格采用固定比例列宽，总宽度适配当前容器，不依赖横向滚动查看右侧字段。
- 目录滚动区的最大高度会随窗口在 `720–1040px` 范围内扩展；大页使用虚拟滚动，只渲染视口附近的行。
- 相关度列只在对应检索启用时显示。
- 支持全局升序/降序和按推荐分、相关度、ID、日期、标题、作者、团队、标签、语言、页数、本地路径排序。
- 可一键复制当前页为制表符分隔文本，也可批量提交当前页封面刷新。
- 点击首列一次即可选择；选中后在当前行下直接展开完整信息，再次点击取消。
- 展开内容包括封面、摘要、作者、团队、语言、页数、日期、基础/推荐分、三种相关度、标题词、完整解析标签、文件名、本地路径和来源链接；原始未解析标签仍只在漫画详情页显示。
- 右侧大图预览可在工具栏中展开或收起，是否显示的偏好会保存在当前浏览器；首次使用时宽屏默认展开，窄于 `1440px` 的窗口默认收起。开启后，鼠标悬停目录行或键盘聚焦行内操作会切换预览，移开时回到当前选中条目。
- 库存主页不再重复显示全局偏好图表，目录表格直接向下占用这部分空间；完整图表仍在“偏好图表”页查看。

选择条目本身不会写入历史；只有打开本地目录或网络来源时才记录。

### 漫画详情、本地目录与封面

详情页同时显示：

- 数据库中的原始未解析标签；
- 经过 `STOP_TAGS.txt` 过滤和 `SEMANTIC_MAP.json` 映射后的解析标签；
- 标题特征词、评分、相关度、文件名、路径和来源。

网络来源统一经过 `/api/track/{id}` 记录历史后重定向。Windows 本地运行或便携版可以调用文件管理器打开漫画目录；Docker 默认关闭该能力，因为容器不能替用户打开宿主机目录。

封面读取优先级为：

1. `b64_cache/<ID>.txt`；
2. `onlineimgtmp` 中的同 ID 图片；
3. 本地漫画目录中的 `1.*`，生成到 `localimgtmp/<ID>.jpg`；
4. 允许联网时尝试在线补抓，并写入 `onlineimgtmp` 与 `b64_cache`。

### 历史与图表

- 默认保留最近 50 次“打开本地目录”或“打开网络来源”记录。
- 源码版和便携版都默认写入项目/发行包根目录的 `datacache/recommendation_history.json`；Docker 容器中对应 `/app/datacache/recommendation_history.json`。
- 支持刷新、单条删除、复选批量删除和二次确认清空。
- 历史中的标签、作者和标题词会参与后续评分。
- 全局和历史图表均提供标签、作者、标题词 Top 15 条形图和可展开 Top 150 数据表。

### LLM 助手

- 支持本地 LM Studio 和线上 OpenAI 兼容 API。
- 在本地/线上模式之间切换后，可直接填写对应的 API URL、模型名和 API Key 并保存；URL 应填写 OpenAI 兼容根地址，服务会追加 `/chat/completions`。
- 保存目标随运行方式自动选择：源码写项目根目录 `.env`，Docker 写宿主项目根目录 `.env`，Windows 便携版写包根目录 `portable-settings.env`。
- 保存后下一次对话立即使用新配置，不需要重启服务；留空 Key 会保留原值，也可以显式清除。
- 已保存的 Key 不会由后端明文回传，界面只显示“已配置/未配置”状态。本机配置接口仅接受通过 `127.0.0.1`、`localhost` 或 `::1` 访问的本机页面请求。
- 可调 Temperature、最大 Tokens 和随机注入条目数量，并可逐轮开启或关闭深度思考模式。
- 每次提问只会从库存主页当前页已经显示的条目中无放回随机抽取 N 条，不会固定取排序前 N 条，也不会在当前页为空时回退到全库抽取。
- 后端通过 Server-Sent Events 分别流式转发真实的思考与回答；支持 `reasoning_content`、`reasoning`，并兼容跨流分块的 `<think>...</think>`。
- 思考开始后会自动展开，回答开始或思考结束后自动折叠，完成后仍可手动重新展开；请求模式、Temperature 等参数单独显示，不会冒充思考内容。
- 官方 DeepSeek API 使用 `thinking.type` 控制思考开关，阿里云百炼兼容接口使用 `enable_thinking`；其他兼容接口不会被强行注入供应商私有参数，但其实际返回的独立思考流仍会显示。
- 用户名称显示为 `YOU`，用户发言与助手回答使用相同正文尺寸；输入、回答和思考过程均支持安全的 Markdown/GFM 渲染，原始 HTML 不会执行。
- 旧对话折叠，最近一次问答保持展开。
- 本轮实际注入的条目列表默认折叠，展开后可通过封面、ID、标题和作者信息返回详情页；条目快照不会因随后切换分页而消失。

聊天和任务输出只保存在当前进程/浏览器状态中，重启后不会恢复。是否能产生独立思考流仍取决于所选模型和接口；未返回 reasoning 字段时，页面不会用请求参数伪造思考过程。

## 一键导入词典与数据

入口：`/admin` → “一键导入词典 / 数据”。

### 支持的文件

- 单个 `.csv`；
- 一个 `.zip`，可在任意层级包含多个 CSV 和标准词典；
- 默认上传上限 `1024 MB`；
- CSV 编码支持 UTF-8 与 UTF-8 BOM；
- ZIP 解压会检查绝对路径、`..` 和目录越界。

标准词典文件名必须完全匹配：

| 文件 | 用途 |
| --- | --- |
| `STOP_TAGS.txt` | 从评分/展示用解析标签中排除噪声标签 |
| `SEMANTIC_MAP.json` | 把多个原始标签聚合到统一语义标签 |
| `TITLE_STOP_WORDS.txt` | 标题分词停用词 |
| `TITLE_SEMANTIC_MAP.json` | 标题词同义映射 |

TXT 既支持单引号包围、逗号分隔的旧格式，也支持每行一个值；空行和以 `#` 开头的行会忽略。JSON 必须是有效的对象映射。

替换词典时使用原子写入，旧文件备份到：

```text
<XP_GACHA_DATA_ROOT>/datacache/imports/backups/<时间戳>/
```

### CSV 列

`链接` 是唯一必需列。标准列如下：

| 列名 | 必需 | 处理规则 |
| --- | --- | --- |
| `ID` | 否 | 空值时尝试从 NH `/g/<id>` 或 JM `/album/<id>` 链接推导 |
| `链接` | 是 | 用于 ID 推导、来源跳转与历史记录 |
| `文件名` | 否 | 可用于匹配本地目录；标题为空时作为标题回退 |
| `标题` | 否 | 空值时使用文件名 |
| `标题译文` | 否 | 增量模式下空值不会覆盖已有译文 |
| `标签` | 否 | 保存数据库原始标签 |
| `作者` | 否 | 用于筛选、评分和图表 |
| `团队` | 否 | 用于搜索和详情 |
| `语言` | 否 | 用于展示和排序 |
| `页数` | 否 | 无法解析时记为 `0` |
| `上传日期` | 否 | 建议先在附录中统一格式 |

无法得到 ID 的行会丢弃；相同 ID 保留最后一条。

### 导入模式

| 模式 | 行为 |
| --- | --- |
| 增量写入 / 更新 | 按 ID 插入或更新；空的标题译文保留数据库已有值 |
| 覆盖重建 | 重建 `gallery_info` 表；界面会再次确认 |

CSV 导入完成后会优化 MySQL 表与全文索引；所有成功导入都会删除 `preprocessed_df.pkl` 与 `data.hash` 并刷新库存，词典导入还会立即重载词典。

“导入项目 `data/gallery_info`”读取的是：

```text
<XP_GACHA_DATA_ROOT>/data/gallery_info/*.csv
```

源码版和便携版都对应项目/发行包根目录的 `data/gallery_info`；Docker 容器中对应 `/app/data/gallery_info`。上传导入使用临时文件，结束后会删除临时副本，请自行保留原始 CSV/ZIP。

## 附录：数据处理与采集

后端只允许执行 26 个白名单任务，同一时间最多运行一个任务。界面会轮询合并输出，支持手动中止；危险操作要求显式勾选确认。

| 分区 | 任务 |
| --- | --- |
| A.1 CSV 整理 | 补全文件名列、NH 链接补 ID、迁移语言标签、清洗上传日期、标题词频统计、聚合未映射标签、语义映射补原名 |
| A.2 数据库同步 | 增量同步 CSV、覆盖重建 MySQL 表、优化表结构与全文索引 |
| A.3 标题 AI 翻译 | LM Studio / 在线兼容 API、批次、并发、范围、重试、JSONL 审查 |
| A.4 缓存与向量 | Base64 预编码、文本语义向量、CLIP 封面向量构建/统计、缓存清理 |
| A.5 维护工具 | 图片/缓存 ID 前缀修正、合并 Base64 增量缓存、清理翻译 JSONL failed 条目、按 ID 删除数据库行、按 `error.json` 清空标题译文 |
| A.6 采集入口 | NH/JM 在线采集、NH/JM 失败页重试、NH 本地链接抓信息、NH 本地链接抓图片 |

系统状态区显示 CSV、线上封面、本地缩略图、Base64、数据库行数，以及四项缓存/向量文件是否存在；模型状态由 `/api/system/status` 返回，但当前附录页面尚未单独展示。

> [!NOTE]
> 表单中的“超时秒数”目前主要作为任务参数保留，通用任务管理器尚未按该值自动结束子进程；需要时请使用“中止任务”。标题翻译自己的单次请求超时仍会生效。

在线采集依赖第三方站点的可访问性和页面结构。请遵守目标站点条款、当地法律和合理请求频率。

## 模型与向量

发行版不会携带模型和向量。默认路径：

| 能力 | 模型 | 索引 |
| --- | --- | --- |
| 文本语义检索 | `models/Qwen3-Embedding-0.6B` | `manga_vectors/manga_vectors_Qwen3.pkl` |
| 封面相似检索 | `models/clip-vit-base-patch32` | `manga_vectors/clip_image_index.pkl` |

可在“附录 → 缓存与向量”中构建索引，也可通过环境变量覆盖路径。模型或索引缺失时，普通关键词、评分、历史和数据库功能仍可使用；对应 AI 检索会给出警告。

## 架构

```mermaid
flowchart LR
    Browser[浏览器 / React SPA] -->|HTTP + SSE| API[FastAPI / server.main]
    API --> Library[LibraryModule<br/>召回・评分・分页]
    API --> History[History / Charts]
    API --> Import[CSV / ZIP Import]
    API --> Jobs[JobsModule<br/>白名单子进程]
    API --> Chat[LM Studio / 在线兼容 API]
    Library --> MySQL[(MySQL gallery_info)]
    Library --> Runtime[词典・缓存・模型・向量・封面]
    Import --> MySQL
    Import --> Runtime
    Jobs --> Legacy[既有 data_get / data_processing / tools]
```

FastAPI 在同一个 `8000` 端口提供 API 和编译后的 `web/dist`。后端模块继续复用经过验证的 `data_pipeline.py` 与 `utils_*.py`，数据任务则通过 `python -m server.job_tasks` 在独立子进程中运行。

### 项目结构

```text
XP-Gacha/
├─ web/                         React + TypeScript 前端
│  ├─ src/
│  └─ dist/                    构建产物
├─ server/                      FastAPI 与服务模块
│  ├─ main.py
│  ├─ job_tasks.py
│  └─ modules/
├─ portable/                    Windows 便携启动器与模板
├─ scripts/
│  ├─ start.ps1 / start.sh
│  ├─ stop.ps1
│  └─ build_portable_release.ps1
├─ tests/                       API、便携启动器与 PowerShell 检查
├─ data_get/                    NH/JM 采集器
├─ data_processing/             CSV、翻译、缓存和向量脚本
├─ tools/                       维护脚本
├─ dictionaries/                四个标准词典
├─ data/gallery_info/           项目 CSV
├─ datacache/                   预处理、历史、导入备份
├─ onlineimgtmp/                线上封面
├─ localimgtmp/                 本地缩略图
├─ b64_cache/ / b64_tmp/        Base64 主缓存与增量缓存
├─ manga_vectors/               文本与封面向量索引
├─ models/                      本地模型
├─ library/                     默认漫画目录
├─ config.py                    无密钥、环境变量驱动的运行配置
├─ launcher.py                  源码单进程入口
├─ app.py                       Legacy Streamlit 入口
├─ Dockerfile
└─ docker-compose.yml
```

## 运行时配置

### 服务与数据库

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `XP_GACHA_ENV` | `development` | `development` 时启用开发 CORS |
| `XP_GACHA_HOST` | `127.0.0.1` | Uvicorn 监听地址；只有在已经增加统一鉴权、限流和网络防护后才应改为 `0.0.0.0` |
| `XP_GACHA_PORT` | `8000` | Web/API 端口 |
| `XP_GACHA_FRONTEND_DIST` | `web/dist` | React 构建产物 |
| `XP_GACHA_ALLOW_OPEN_LOCAL` | `true` | 是否允许 Windows 服务端打开本地目录；Docker 强制为 `false` |
| `XP_GACHA_IMPORT_MAX_MB` | `1024` | 导入上传上限 |
| `DATABASE_URL` | 空 | 完整 SQLAlchemy URL，优先级最高 |
| `MYSQL_HOST` | `127.0.0.1` | 数据库主机 |
| `MYSQL_PORT` | `3306` | 数据库端口 |
| `MYSQL_DATABASE` | `xp_gacha` | 数据库名 |
| `MYSQL_USER` | `xp_gacha` | 数据库用户 |
| `MYSQL_PASSWORD` | `xp_gacha` | 数据库密码 |

### 数据、检索与模型

下表中由 `runtime_path` 管理的缓存、词典、模型和向量相对路径都以 `XP_GACHA_DATA_ROOT` 为基准。`XP_GACHA_BASE_DIR` 是例外：显式传入相对路径时会按当前工作目录解析；Docker 给它传入 `/library`，便携启动器也会预先转换为绝对路径。

| 环境变量 | 默认值 |
| --- | --- |
| `XP_GACHA_DATA_ROOT` | 源码项目根目录；Docker `/app`；便携版发行包根目录 |
| `XP_GACHA_BASE_DIR` | `<DATA_ROOT>/library` |
| `ONLINE_IMG_DIR` | `onlineimgtmp` |
| `IMG_CACHE_DIR` | `localimgtmp` |
| `CACHE_DIR` | `datacache` |
| `B64_CACHE_DIR` | `b64_cache` |
| `MODEL_DIR` | `models` |
| `DICTIONARY_DIR` | `dictionaries` |
| `VECTOR_FILE` | `manga_vectors/manga_vectors_Qwen3.pkl` |
| `IMG_VECTOR_FILE` | `manga_vectors/clip_image_index.pkl` |
| `LOCAL_MODEL_PATH` | `models/Qwen3-Embedding-0.6B` |
| `CLIP_MODEL_PATH` | `models/clip-vit-base-patch32` |
| `MAX_DISPLAY` | `500` |
| `SEMANTIC_SEARCH_TOP_K` | `5000` |
| `COVER_SEARCH_TOP_K` | `5000` |
| `HISTORY_RECOMMENDATION_CACHE_SIZE` | `50` |
| `HISTORY_CACHE_FILE` | `datacache/recommendation_history.json` |

### LLM、在线封面与采集

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `LM_STUDIO_API_BASE` | `http://127.0.0.1:1234/v1` | 本地兼容接口 |
| `LM_STUDIO_API_KEY` | 空 | 本地 API 密钥，可留空 |
| `LM_STUDIO_MODEL` | `local-model` | 本地模型 |
| `ONLINE_API_BASE` | 空 | 在线兼容接口 |
| `ONLINE_API_KEY` | 空 | 在线 API 密钥 |
| `ONLINE_MODEL` | `deepseek-v4-flash` | 在线模型 |
| `SYSTEM_PROMPT` | 内置中文提示 | 助手系统提示 |
| `ONLINE_COVER_FETCH_ENABLED` | `true` | 是否允许在线补抓 |
| `ONLINE_COVER_PROXY` | 空 | 在线封面与 NH 采集代理，空值直连；Docker 访问宿主机代理使用 `host.docker.internal` |
| `ONLINE_COVER_FETCH_CONCURRENCY` | `6` | 后台封面并发数 |

不要把含密钥的 `.env`、`portable-settings.env` 或 `config/portable.json` 提交到版本库或发给他人。

## 数据目录、备份与便携版边界

源码和 Docker 的主要数据目录包括：

| 目录 | 内容 |
| --- | --- |
| `data/gallery_info` | 可重复导入的原始 CSV |
| `datacache` | 预处理缓存、历史记录、导入临时目录和词典备份 |
| `dictionaries` | 标签与标题词典 |
| `onlineimgtmp` | 在线封面 |
| `localimgtmp` | 本地目录生成的缩略图 |
| `b64_cache` / `b64_tmp` | Base64 缓存 |
| `manga_vectors` | 文本、封面向量索引 |
| `models` | 本地模型与模型缓存 |
| `library` | 漫画目录 |
| `logs` | 采集和运行日志 |

Docker 的 MySQL 数据位于命名卷 `xp-gacha_mysql-data`，不在项目目录 bind mount 中。备份 Docker 环境时要同时备份数据库卷和项目数据目录。

便携版 v0.2.3 起不再使用 `userdata`。它与源码版使用同一套根目录相对路径，业务数据、缓存、模型和便携版运行数据都在发行包根目录并列存放：

```text
XP-Gacha-v<version>-portable-win64/
├─ config/portable.json
├─ mysql/data/
├─ data/
├─ datacache/
├─ dictionaries/
├─ onlineimgtmp/
├─ localimgtmp/
├─ b64_cache/ / b64_tmp/
├─ manga_vectors/
├─ models/
├─ library/
├─ logs/
├─ run/
└─ tmp/
```

附录任务的默认相对路径现在在源码版和便携版中完全一致。例如直接填写 `data/gallery_info`、`models/Qwen3-Embedding-0.6B`、`manga_vectors/manga_vectors_Qwen3.pkl` 或 `b64_cache`，均会解析到项目/发行包根目录。自定义到包外的绝对漫画目录仍需单独备份。

## 更新 Windows 便携包

当前没有联网自动更新器、自动数据库迁移器或失败自动回滚。采用“新目录解压 + 数据迁移 + 保留旧版回滚”的手动升级流程。

### 从 v0.2.2 迁移到 v0.2.3

v0.2.2 的数据位于 `userdata`，v0.2.3 起改为发行包根目录并列存放。迁移前必须在旧版运行 `Stop XP-Gacha.cmd`，确认 XP-Gacha 和 MySQL 已完全停止，再备份旧包整个 `userdata` 和 `portable-settings.env`。不要直接覆盖旧版目录，应把 v0.2.3 ZIP 解压到新目录。

将下列旧版目录逐一复制到 v0.2.3 新包根目录的同名位置：

| v0.2.2 源路径 | v0.2.3 目标路径 | 说明 |
| --- | --- | --- |
| `userdata/data` | `data` | CSV 和本地输入数据 |
| `userdata/datacache` | `datacache` | 预处理缓存、历史、导入备份 |
| `userdata/b64_cache` | `b64_cache` | Base64 封面缓存 |
| `userdata/b64_tmp` | `b64_tmp` | Base64 增量缓存 |
| `userdata/localimgtmp` | `localimgtmp` | 本地缩略图 |
| `userdata/onlineimgtmp` | `onlineimgtmp` | 在线封面 |
| `userdata/library` | `library` | 默认漫画目录 |
| `userdata/manga_vectors` | `manga_vectors` | 文本与封面向量 |
| `userdata/models` | `models` | 本地模型与模型缓存 |
| `userdata/mysql` | `mysql` | 包内 MySQL 数据；必须与 `config` 成套迁移 |
| `userdata/config` | `config` | 包含 MySQL 随机凭据；必须与 `mysql` 成套迁移 |
| `userdata/logs` | `logs` | 运行和 MySQL 日志，可选 |
| `userdata/cache` | `models/cache/xdg` | 旧版其他工具缓存，可选；通常可不迁移 |
| `userdata/dictionaries/*` | `dictionaries/` | 将自定义词典文件复制到新包的根词典目录 |

`userdata/run` 和 `userdata/tmp` 是临时目录，不要迁移。若旧包根目录已有附录任务产生的 `data`、`models`、`manga_vectors`、`b64_cache` 或其他输出，也要在停机后与对应新目录仔细合并。`data_processing`、`tools` 等同时包含程序代码的目录只能复制用户生成文件，不能整个覆盖新版代码。

> [!CAUTION]
> `mysql` 与 `config` 必须来自同一份完整备份并成套迁移。绝不能只复制 `mysql/data`、只复制 `config/portable.json`，或把两个不同实例的目录混用；启动器检测到只存在其中一项时会直接中止，避免静默初始化空库。MySQL 正在运行时也绝不能复制其原始数据目录。若新旧 `BUILD-INFO.json` 中的 `runtime.mysql.version` 不同，不保证原始 `mysql/data` 可直接迁移；应改用逻辑导出/导入或等待对应迁移说明。

逐项把旧设置合并进新版 `portable-settings.env`，启动新版后核对库存、历史、词典、封面、模型和漫画目录。保留旧版目录作为回滚，确认稳定后再删除。

v0.2.3 之后在两个根目录并列布局版本之间升级时，仍要先停机并完整备份所有数据目录；`mysql` 和 `config` 始终必须作为一个整体备份与迁移。

## 构建 Windows 便携发行版

版本号从 `server/__init__.py` 读取，`web/package.json` 的版本只是前端包内部版本，不决定发行包名称。

构建机需要：

- Windows x64 PowerShell；
- CPython 3.13 x64 且有 pip；
- Node.js 与 pnpm；
- Git；
- Windows 自带的 `curl.exe`、`tar.exe`、`robocopy.exe`、`expand.exe`；
- 网络连接和数 GB 可用空间。

执行：

```powershell
.\scripts\build_portable_release.ps1
```

默认输出目录相对于项目根目录解析：

```text
..\XP-Gacha-Releases
```

它不是写死的盘符路径。默认产物：

```text
..\XP-Gacha-Releases/
├─ XP-Gacha-v<version>-portable-win64/
├─ XP-Gacha-v<version>-portable-win64.zip
└─ XP-Gacha-v<version>-portable-win64.zip.sha256
```

构建流程会：

1. 构建 React 前端；
2. 按白名单复制程序，排除业务数据和开发缓存；
3. 下载并校验固定版本的 Python、MySQL、WiX，以及微软签名有效的当前 VC++ 运行库；
4. 将 CPU PyTorch 与全部 Python 依赖安装到嵌入式 Python；
5. 执行依赖自检；
6. 对空 MySQL 数据库执行首次启动 smoke test；
7. 使用已生成的数据库凭据执行第二次重启 smoke test；
8. 清空验证产生的 MySQL、配置、日志、缓存等根目录运行数据；
9. 审计发行包不含业务数据；
10. 生成 `BUILD-INFO.json`、`requirements-lock.txt`、逐文件 SHA-256、ZIP 和 ZIP 校验值。

常用参数：

```powershell
# 指定相对于项目根目录的输出目录
.\scripts\build_portable_release.ps1 -OutputRoot ".\releases"

# 指定 CPython 3.13 x64
.\scripts\build_portable_release.ps1 -BuildPython "C:\path\to\python.exe"

# 复用已经确认最新的 web/dist
.\scripts\build_portable_release.ps1 -SkipFrontendBuild
```

正式发布不要使用 `-SkipVerification`；否则 `BUILD-INFO.json` 会记录首次启动未验证。下载和 pip 缓存位于项目的 `.portable-cache`，不会进入发行包。

根目录 `dictionaries` 中四个基础词典文件属于程序默认配置并纳入版本控制；构建只把这四个默认词典作为初始内容带入无数据发行包，漫画、数据库、历史、模型、向量和缓存仍不会打包。

`-Force` 会递归删除输出目录中精确同版本的发行目录、ZIP 和校验文件。不要对已经含有真实 `data`、`mysql`、`config`、模型、封面或其他用户数据的活动发行目录使用它。推荐先更新 `server/__init__.py` 的版本号，再构建一个新的版本目录。

推荐发布顺序：

1. 完成功能和文档；
2. 更新 `server/__init__.py`；
3. 运行前端构建和测试；
4. 保持 Git 工作区干净并提交；
5. 运行默认的完整便携包构建；
6. 核对 `BUILD-INFO.json`、ZIP `.sha256` 和 `sourceDirty`；
7. 发布 ZIP 与对应校验文件。

## API 概览

完整请求模型和在线调试请查看 `/api/docs`。

| 分类 | 方法与路径 |
| --- | --- |
| 健康/系统 | `GET /api/health`、`GET /api/system/status` |
| 库存选项 | `GET /api/meta/options`、`GET /api/meta/options/search` |
| 库存/详情 | `POST /api/library/query`、`GET /api/gallery/{id}` |
| 封面 | `GET /api/covers/{id}`、`GET /api/covers/status`、`POST /api/covers/refresh`、`POST /api/search/cover` |
| 历史 | `GET/POST/DELETE /api/history`、`DELETE /api/history/all` |
| 打开/跳转 | `POST /api/gallery/{id}/open-local`、`GET /api/track/{id}` |
| 图表 | `GET /api/charts/global`、`GET /api/charts/history` |
| LLM | `POST /api/chat/stream`、`GET/PUT /api/chat/settings` |
| 任务 | `GET /api/scripts`、`POST /api/jobs`、`GET /api/jobs/{id}`、`POST /api/jobs/{id}/cancel` |
| 导入 | `POST /api/import/bundle`、`POST /api/import/project` |
| 偏好 | `GET/PUT /api/preferences` |

`/api/preferences` 当前只提供列宽 JSON 的后端读写兼容接口，React 表格没有接入可视化列宽编辑，实际使用固定比例列宽。

`/api/chat/settings` 是仅供助手页使用的本机配置管理接口：它拒绝非回环地址，写请求还需要同源页面校验标记；响应只包含 URL、模型名和 Key 是否已配置，不包含 Key 明文。

## 测试与校验

后端/API 与便携启动器：

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

前端类型检查和生产构建：

```powershell
pnpm --dir web typecheck
pnpm --dir web build
```

Windows PowerShell 脚本检查：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tests\check_windows_powershell_scripts.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File tests\check_start_script_failure.ps1
```

常用运行检查：

```powershell
docker compose ps
docker compose logs --tail 200 app mysql
```

## 故障排查

### `Docker was not found`

确认 Docker Desktop / Docker Engine 已安装且正在运行：

```powershell
docker version
docker compose version
```

Windows 使用 WSL 2 后端时，还需要系统虚拟化、WSL 和 Linux 发行版处于可用状态。完成这些系统变更后通常需要重启。

### Docker 启动后网页打不开

```powershell
docker compose ps
docker compose logs --tail 200 app mysql
```

确认 `.env` 中的 Web/MySQL 端口没有被占用，并检查 `http://127.0.0.1:<XP_GACHA_PORT>/api/health`；未改配置时端口为 `8000`。第一次镜像构建会下载 CPU AI 依赖，耗时和磁盘占用都明显高于普通 Web 项目。

### NH 在线采集提示 `curl (7) ... over proxy 127.0.0.1`

新版 NH 采集器不再强制使用 `127.0.0.1:7890`，而是统一读取 `ONLINE_COVER_PROXY`。留空时明确直连；源码和便携版连接本机代理可填 `http://127.0.0.1:7890`。Docker 容器中的 `127.0.0.1` 指向容器自身，连接 Windows 宿主机代理必须填：

```dotenv
ONLINE_COVER_PROXY=http://host.docker.internal:7890
```

修改 `.env` 或更新采集器代码后，执行 `docker compose up -d --build app` 重建应用容器。若所有列表页都请求失败，任务现在会标记为失败，不再以 `[JOB] completed` 掩盖网络故障。

### 库存为空

这是无数据发行版的正常状态。进入 `/admin` 上传 CSV/ZIP，或把 CSV 放入 `<XP_GACHA_DATA_ROOT>/data/gallery_info` 后点击“导入项目 data/gallery_info”。

### 关键词搜索只有 `LIKE`

先在附录运行“优化 MySQL 表结构与全文索引”。SQLite 等替代数据库不提供与 MySQL `ngram`、表优化和 upsert 完全相同的体验。

### 语义或封面相似检索不可用

检查模型和索引是否存在：

```text
models/Qwen3-Embedding-0.6B
manga_vectors/manga_vectors_Qwen3.pkl
models/clip-vit-base-patch32
manga_vectors/clip_image_index.pkl
```

需要时在“附录 → 缓存与向量”构建。源码版和便携版都直接使用 `models/...` 和 `manga_vectors/...` 路径。

### 便携版浏览器没有自动打开

查看启动窗口打印的实际 URL。端口冲突时启动器会自动更换端口，也可以手动编辑 `portable-settings.env` 的首选端口。

### 便携版 MySQL 或应用启动失败

依次查看：

```text
logs/mysql-initialize.log
logs/mysql-error.log
logs/mysql-console.log
logs/app.log
```

如果提示“MySQL 数据目录不完整且非空”，先停止程序并成套备份根目录 `mysql` 与 `config`，不要直接删除。只有确认从未导入数据且允许完全重建时，才考虑清理旧数据。

### `ERROR 1045 ... using password: NO`

先查看 `BUILD-INFO.json`，确认使用的是当前 `v0.2.5`，不是旧的 `v0.2.0` 启动器。`v0.2.1` 及以后版本已加入 MySQL 8.4 公钥认证参数。

同时确认：

- 根目录 `mysql` 与 `config/portable.json` 来自同一份完整备份；
- 没有单独删除或重建 `portable.json`；
- 旧版程序已经完全停止。

### 安全软件提示便携包

先核对 ZIP 同目录 `.sha256` 和包内 `SHA256SUMS.txt`，再重新解压。不要通过永久关闭系统安全功能解决。分享日志前请移除私人路径，绝不能分享 `portable.json` 或带密钥的 `portable-settings.env`。

## 当前限制

- 不附带漫画目录、CSV、业务数据库、历史、封面、模型或向量；便携包只携带程序默认词典。
- 没有登录鉴权，不适合直接公网部署。
- 本地目录打开仅适用于 Windows 桌面运行；Docker 中默认关闭。
- LLM 助手依赖 LM Studio 或用户自己的兼容 API。
- 语义/封面检索依赖用户提供模型和索引。
- 在线封面和采集依赖网络及第三方站点状态。
- 同时只能运行一个附录任务。
- 任务状态、终端输出和聊天记录不持久化，服务重启后丢失。
- 通用任务“超时秒数”尚未由任务管理器强制执行。
- 便携版当前没有自动更新器，升级采用手动迁移。

## Legacy Streamlit

`app.py` 保留旧 Streamlit 界面以兼容已有工作流：

```powershell
streamlit run app.py
```

它不是当前主入口，不包含 React 表格虚拟化、固定比例布局、同端口 `/api/track` 等新版交互。旧版还可能读取 `.streamlit/secrets.toml` 和使用独立 `8765` 链接追踪服务；这些都不应作为新部署方案。

仓库 `UI-imgs` 中现有图片属于旧 Streamlit 界面，因此本 README 不再把它们作为当前 Web 截图。

## 数据与使用责任

请只导入、整理和访问你有权处理的内容。在线采集功能应遵守目标站点条款、robots 策略、版权要求和所在地法律；项目不会替用户提供、分发或授权漫画数据。

## License

本项目采用 [MIT License](LICENSE)。
