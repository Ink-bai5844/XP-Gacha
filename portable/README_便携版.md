# XP-Gacha Windows x64 便携版

这是无需安装 Python、Node.js、Docker 或 MySQL 的完整便携发行版。Python、CPU 版 AI 依赖和 MySQL 8.4 都位于当前文件夹的 `runtime` 中；启动器不会修改系统 PATH、注册 Windows 服务或向全局 Python 安装包。

## 一键启动

1. 完整解压 ZIP。不要直接在压缩软件预览窗口中运行。
2. 双击 `Start XP-Gacha.cmd`。
3. 首次启动会在发行包根目录初始化空 MySQL 数据库和各数据目录，通常需要几十秒；完成后自动打开浏览器。
4. 打开“附录 → 一键导入”，上传 CSV 或 ZIP。发行版不附带漫画目录、馆藏数据库、图片缓存、模型或个人历史。

启动窗口需要保持开启。正常停止请按 `Ctrl+C`，或双击 `Stop XP-Gacha.cmd`。再次启动后，数据库和配置会继续使用当前发行包根目录下的并列数据目录。

## 常用文件

- `Start XP-Gacha.cmd`：启动数据库、后端与网页并打开浏览器。
- `Stop XP-Gacha.cmd`：安全停止当前这一份发行包的进程。
- `Check XP-Gacha.cmd`：检查内置运行时和全部 Python 功能依赖。
- `Update XP-Gacha.cmd`：在线检查最新正式版，并安全安装兼容的应用层增量更新。
- `Open XP-Gacha Folder.cmd`：打开发行包根目录，数据库、缓存、日志和导入目录都在这里并列存放。
- `portable-settings.env`：可选端口、漫画目录、LLM/API、在线封面与 NH 采集代理配置；不修改也能启动。
- `logs`：启动失败时查看 `app.log`、`mysql-error.log` 和初始化日志。

## 助手 API 配置

打开“助手”页，选择“本地 (LM Studio)”或“线上 API”，即可填写对应的 OpenAI 兼容 API URL、模型名和 API Key。点击保存后，配置会写入当前发行包根目录的 `portable-settings.env`，下一次对话立即生效，不需要重启程序。

为了避免泄露密钥，后端不会把已保存的 Key 明文传回页面，只会显示是否已经配置。Key 输入框留空会保留原值；需要删除时，请点击“清除 Key”后再保存。本地 LM Studio 无需鉴权时，`LM_STUDIO_API_KEY` 可以留空。配置管理接口仅接受通过 `127.0.0.1`、`localhost` 或 `::1` 打开的本机页面请求。

`portable-settings.env` 含有 Key 时不要分享、提交或随日志一起发送。一键增量更新不会覆盖该文件；改用完整新版 ZIP 手动迁移时，应把其中的自定义项逐项合并到新包，而不是用旧文件整体覆盖新版模板。

## 数据与模型

发行包内只附带根目录 `dictionaries` 中的基础词典规则，之后可在界面中用 ZIP 一键替换。漫画 CSV、MySQL 业务表、封面、历史、向量和模型均为空。

如果使用本地语义/封面模型，请把模型和向量放入根目录的 `models` 与 `manga_vectors`，或在附录任务中按界面默认路径生成。AI 对话仍需要你自己的 LM Studio 或线上兼容 API；采集、在线封面和线上 API 功能仍需要网络。这些是项目功能本身的外部数据/服务，不属于需要安装的运行环境。

便携版与源码版现在使用同一套根目录相对路径：`data`、`datacache`、`b64_cache`、`b64_tmp`、`localimgtmp`、`onlineimgtmp`、`library`、`manga_vectors`、`models`、`mysql`、`config`、`run`、`logs` 和 `tmp` 均与程序文件并列；词典位于根目录 `dictionaries`，工具缓存位于 `models/cache`。`updates` 仅用于更新锁、下载暂存和 `updates/backups` 中的程序文件回滚备份。

## 一键在线检查与增量更新

双击根目录的 `Update XP-Gacha.cmd` 即可检查 GitHub 上最新的正式 Release，并在有兼容更新时完成下载、校验、停机、替换、自检和按需重启。只检查版本、不安装时，在 PowerShell 或命令提示符中运行：

```powershell
.\Update XP-Gacha.cmd -CheckOnly
```

更新包只包含应用程序层文件，不包含包内 Python、MySQL 或 VC++ 运行时。安装前会校验 Release 清单、更新 ZIP 和逐文件 SHA-256，并确认当前 Python、MySQL 和 `requirements-lock.txt` 与目标版本兼容；不兼容时会保持当前版本不变并提示下载完整便携 ZIP。

下列内容始终受保护，不会被增量更新覆盖或删除：

- `runtime`、`mysql`、`config`、`run`、`logs` 和 `tmp`；
- `data`、`datacache`、`library`、`models`、`manga_vectors`、封面与缓存目录；
- `dictionaries`、`portable-settings.env` 以及目录中的其他未知用户文件。

更新器只在受控范围内整体替换 `web/dist`，其他程序文件按哈希仅复制有变化的内容。变更前的程序文件保存在 `updates/backups`；安装失败或安装后自检失败时会自动回滚。它不执行数据库或运行时迁移，因此看到兼容性提示时，应按 Release 说明解压完整新版到新目录并手动迁移数据。

> [!IMPORTANT]
> `v0.2.6` 及更早版本没有一键更新入口。请先下载 `v0.2.7` 或更高版本的完整便携 ZIP，并按下方说明迁移一次；以后即可直接使用同一个 `Update XP-Gacha.cmd`。

## 从 v0.2.2 升级到 v0.2.3

v0.2.2 把数据放在 `userdata`，v0.2.3 起改为发行包根目录下的并列目录。升级时请先在旧版执行 `Stop XP-Gacha.cmd`，确认 MySQL 已完全停止，并备份整个 `userdata`。将旧版以下目录逐一复制到 v0.2.3 新包根目录的同名位置：

```text
userdata/data             -> data
userdata/datacache        -> datacache
userdata/b64_cache        -> b64_cache
userdata/b64_tmp          -> b64_tmp
userdata/localimgtmp      -> localimgtmp
userdata/onlineimgtmp     -> onlineimgtmp
userdata/library          -> library
userdata/manga_vectors    -> manga_vectors
userdata/models           -> models
userdata/mysql            -> mysql
userdata/config           -> config
userdata/logs             -> logs
userdata/cache            -> models/cache/xdg
userdata/dictionaries/*   -> dictionaries/
```

还要单独检查旧版发行包根目录，不要只备份 `userdata`。v0.2.2 及更早版本的部分附录任务可能已经按当时的相对路径，把任务结果直接写入旧包根目录的 `data`、`models`、`manga_vectors`、`b64_cache` 等目录，或写成 `data_processing/*.jsonl` 等任务产物。这些根目录数据与上表中 `userdata` 里的数据都可能是有效的，需要按目标位置合并迁移：

```text
旧包根目录/data/*                + userdata/data/*          -> 新包/data/
旧包根目录/models/*              + userdata/models/*        -> 新包/models/
旧包根目录/manga_vectors/*       + userdata/manga_vectors/* -> 新包/manga_vectors/
旧包根目录/b64_cache/*           + userdata/b64_cache/*     -> 新包/b64_cache/
旧包根目录/data_processing/*.jsonl                           -> 新包/data_processing/
```

最稳妥的做法是在新包首次启动前完成合并。如果新包已经产生数据，请先备份新旧两份内容，再只补充缺失文件；遇到同名文件时先根据生成时间和内容完整性人工确认，不要用旧目录整体覆盖新数据。`data_processing` 只迁移附录任务生成的 `.jsonl` 等数据文件，不要整个覆盖新版脚本目录。

`run` 和 `tmp` 是临时目录，不要从旧包迁移。`mysql` 与 `config` 必须来自同一份完整备份并成套迁移，绝不能只复制 `mysql/data` 或只复制 `config/portable.json`；启动器会在发现只存在其中一项时中止启动，防止用错误凭据初始化出一套看似正常的空库。若新旧包内 MySQL 版本不同，不要直接复制原始数据目录，请改用逻辑导出/导入或等待对应迁移说明。`portable-settings.env` 中的自定义项仍需要逐项合并到新包。

## 便携性说明

- 正式支持 64 位 Windows 11；MySQL 8.4 当前官方支持矩阵已不再列出 Windows 10。
- 默认只监听 `127.0.0.1`，不会向局域网开放。
- 默认端口为网页 `8000`、MySQL `3307`；冲突时自动选择其他本机端口。
- 可以复制整个文件夹制作互不共享数据库的独立实例。
- 请把发行版放在普通可写目录，不要放入 `Program Files`、只读介质或需要管理员权限的位置。
- 一键增量更新会自动停机并备份本次变更的程序文件；遇到运行时/数据库迁移或改用完整 ZIP 时，仍应先手动停机并整体备份上述数据目录、`dictionaries`、`portable-settings.env`，其中 `mysql` 与 `config` 必须成套备份。
