# XP-Gacha Windows x64 便携版

这是无需安装 Python、Node.js、Docker 或 MySQL 的完整便携发行版。Python、CPU 版 AI 依赖和 MySQL 8.4 都位于当前文件夹的 `runtime` 中；启动器不会修改系统 PATH、注册 Windows 服务或向全局 Python 安装包。

## 一键启动

1. 完整解压 ZIP。不要直接在压缩软件预览窗口中运行。
2. 双击 `Start XP-Gacha.cmd`。
3. 首次启动会在 `userdata` 中初始化空 MySQL 数据库，通常需要几十秒；完成后自动打开浏览器。
4. 打开“附录 → 一键导入”，上传 CSV 或 ZIP。发行版不附带漫画目录、馆藏数据库、图片缓存、模型或个人历史。

启动窗口需要保持开启。正常停止请按 `Ctrl+C`，或双击 `Stop XP-Gacha.cmd`。再次启动后，数据库和配置会继续使用当前文件夹中的 `userdata`。

## 常用文件

- `Start XP-Gacha.cmd`：启动数据库、后端与网页并打开浏览器。
- `Stop XP-Gacha.cmd`：安全停止当前这一份发行包的进程。
- `Check XP-Gacha.cmd`：检查内置运行时和全部 Python 功能依赖。
- `Open User Data.cmd`：打开数据库、缓存、日志和导入目录。
- `portable-settings.env`：可选端口、漫画目录、LLM/API 与在线封面配置；不修改也能启动。
- `userdata/logs`：启动失败时查看 `app.log`、`mysql-error.log` 和初始化日志。

## 数据与模型

发行包内只附带程序使用的基础词典规则；首次运行会复制到 `userdata/dictionaries`，之后可在界面中用 ZIP 一键替换。漫画 CSV、MySQL 业务表、封面、历史、向量和模型均为空。

如果使用本地语义/封面模型，请把模型和向量放入 `userdata/models` 与 `userdata/manga_vectors`，或在附录任务中按界面路径生成。AI 对话仍需要你自己的 LM Studio 或线上兼容 API；采集、在线封面和线上 API 功能仍需要网络。这些是原项目功能本身的外部数据/服务，不属于需要安装的运行环境。

## 便携性说明

- 正式支持 64 位 Windows 11；MySQL 8.4 当前官方支持矩阵已不再列出 Windows 10。
- 默认只监听 `127.0.0.1`，不会向局域网开放。
- 默认端口为网页 `8000`、MySQL `3307`；冲突时自动选择其他本机端口。
- 可以复制整个文件夹制作互不共享数据库的独立实例。
- 请把发行版放在普通可写目录，不要放入 `Program Files`、只读介质或需要管理员权限的位置。
- 更新前先停止程序并备份 `userdata`；不要用新版空目录覆盖自己的 `userdata`。
