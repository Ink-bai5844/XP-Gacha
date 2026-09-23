# XP-Gacha macOS 一键便携版

本包面向 **macOS 15 或更新版本、原生 Apple Silicon（arm64）**。Intel Mac 和 Rosetta 模式不受支持；不要勾选终端应用的“使用 Rosetta 打开”。

完整包内置 Python standalone 3.12.14、MySQL Community Server 8.4.11、Python 依赖和已构建网页。运行时无需安装 Python、Node.js、MySQL、Homebrew 或 Docker。漫画、业务数据、历史记录、封面、模型与向量不随包提供；只附带程序默认词典。

## 解压与启动

1. 下载完整 `XP-Gacha-v<version>-portable-macos-arm64.tar.gz` 及同名 `.sha256`，核对下载校验值。
2. 完整解压到普通本地可写目录。使用 macOS 归档实用工具，或运行 `tar -xzf "XP-Gacha-v<version>-portable-macos-arm64.tar.gz"`，保留可执行权限和符号链接。不要在压缩预览窗口内启动，也不要把包放在只读目录。
3. 双击 `Start XP-Gacha.command`。启动器初始化包内 MySQL、创建随机数据库凭据，启动后打开浏览器。首次初始化可能需要等待。
4. 保持启动终端窗口开启；停止时按 `Ctrl+C`，或双击 `Stop XP-Gacha.command`。
5. 首次使用按库存首页提示下载数据 ZIP，再在“附录 → 一键导入词典 / 数据”中导入。AI 检索需要另外准备对应模型与向量。

网页首选端口为 `8000`，MySQL 首选端口为 `3307`；占用时会选择其他可用端口。以启动窗口打印的网址为准。程序仅监听本机，不安装系统服务，不修改全局 Python、MySQL 或 `PATH`。

本发行包没有进行 Apple 代码签名或公证。首次启动可能被 macOS 拦截：确认来源并核对校验值后，在“系统设置 → 隐私与安全性”中查看对应拦截，按系统提示对这个包允许打开。系统可能分别检查包内 Python、MySQL 等可执行文件。不要关闭 Gatekeeper，不要批量移除隔离属性。若系统报告文件损坏或无法确认来源，应重新核对来源与下载完整性。

## 包内入口与设置

| 文件 | 用途 |
| --- | --- |
| `Start XP-Gacha.command` | 启动本包的数据库、API 和网页 |
| `Stop XP-Gacha.command` | 停止本包实例 |
| `Check XP-Gacha.command` | 检查运行文件与 Python 依赖；检查完保留终端窗口 |
| `Open XP-Gacha Folder.command` | 在 Finder 中打开本包目录 |
| `portable-settings.env` | 设置端口、漫画目录、LLM 和代理等 |
| `BUILD-INFO.json` | 运行时版本、源码版本及本次构建的验证结果 |
| `requirements-lock.txt` | 实际随包安装的 Python 依赖版本 |
| `SHA256SUMS.txt` | 刚解压发行目录的逐文件校验值 |

`Check` 不启动 MySQL，不能替代首次启动验证。完整包的 `.sha256` 用于核对压缩包，包内清单用于核对初始文件；实际运行后产生数据库与缓存、保存配置或修改词典，会使相应文件发生正常变化。

压缩包可使用 `shasum -a 256 -c "XP-Gacha-v<version>-portable-macos-arm64.tar.gz.sha256"` 核对。包内 `SHA256SUMS.txt` 对普通文件计算内容哈希，对符号链接计算其 UTF-8 链接目标文本的哈希，因此不能直接用 `shasum -c` 校验整份包内清单。

编辑 `portable-settings.env` 中的 `XP_GACHA_LIBRARY_PATH` 指向漫画目录。相对路径以发行包根目录为基准，也可以使用包含空格、中文的绝对路径。助手页保存的 LLM 设置也写入这份文件；它可能包含 API 密钥，请勿公开。网页打开本地漫画目录时会调用 Finder。

## 数据、备份与升级

所有运行数据放在包根目录的 `data`、`datacache`、`mysql`、`config`、`logs`、`models`、`manga_vectors`、封面与缓存等目录。开始备份前先正常停止实例。`mysql` 与 `config` 必须来自同一份完整备份并成套保存；`config/portable.json` 包含随机数据库密码，不要分享。

macOS 包目前没有自动增量更新入口。升级时解压完整新版到新目录，先停止旧版并备份，再迁移业务数据和个人设置，保留旧目录便于回退。不要覆盖正在运行的包，不迁移 `run`、`tmp` 等临时目录。运行时版本不一致或跨操作系统迁移时，应使用数据库逻辑导出与导入，不直接复制 MySQL 原始数据目录。

是否通过完整启动与重启验证，以此包的 `BUILD-INFO.json` 为准。构建时使用 `--skip-verification` 的产物不能视为完成正式发布验证。
