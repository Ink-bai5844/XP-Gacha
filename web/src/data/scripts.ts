export type ScriptFieldValue = string | number | boolean | string[];

export type ScriptField = {
  id: string;
  label: string;
  type: "text" | "password" | "number" | "checkbox" | "select" | "multiselect" | "textarea";
  defaultValue: ScriptFieldValue;
  min?: number;
  max?: number;
  step?: number;
  options?: string[];
  help?: string;
  disabledWhen?: { field: string; equals: ScriptFieldValue };
};

export type ScriptDefinition = {
  id: string;
  title: string;
  action: string;
  description: string;
  defaultOpen?: boolean;
  confirmField?: string;
  fields: ScriptField[];
};

export type ScriptSection = {
  id: string;
  code: string;
  title: string;
  scripts: ScriptDefinition[];
};

const text = (id: string, label: string, defaultValue = "", help?: string): ScriptField => ({ id, label, type: "text", defaultValue, help });
const password = (id: string, label: string, defaultValue = "", help?: string): ScriptField => ({ id, label, type: "password", defaultValue, help });
const textarea = (id: string, label: string, defaultValue = "", help?: string): ScriptField => ({ id, label, type: "textarea", defaultValue, help });
const number = (id: string, label: string, defaultValue: number, min: number, max: number, step: number): ScriptField => ({ id, label, type: "number", defaultValue, min, max, step });
const checkbox = (id: string, label: string, defaultValue = false, help?: string): ScriptField => ({ id, label, type: "checkbox", defaultValue, help });
const select = (id: string, label: string, defaultValue: string, options: string[]): ScriptField => ({ id, label, type: "select", defaultValue, options });
const multiselect = (id: string, label: string, defaultValue: string[], options: string[]): ScriptField => ({ id, label, type: "multiselect", defaultValue, options });
const timeout = (value = 600, max = 7200): ScriptField => number("timeout", "超时秒数", value, 10, max, 10);

export const dataSections: ScriptSection[] = [
  {
    id: "csv",
    code: "A.1",
    title: "CSV 整理",
    scripts: [
      { id: "addname", title: "补全文件名列", action: "执行补全", description: "依据本地链接 HTML/TXT 为 CSV 补充文件名并写入新文件。", fields: [text("csvFile", "CSV 文件", "data/gallery_info_no_name/JM_info_yuri.csv"), text("txtFile", "本地链接 HTML/TXT", "data/local_data/NH_all.txt"), text("outputFile", "输出 CSV", "data/gallery_info/JM_info_yuri_full.csv"), timeout()] },
      { id: "add-id", title: "NH 链接补 ID", action: "补齐 ID", description: "扫描 CSV 目录，并按指定前缀补齐缺失的唯一 ID。", fields: [text("csvDir", "CSV 目录", "data/gallery_info_no_name"), text("prefix", "ID 前缀", "NH"), timeout()] },
      { id: "add-lang", title: "迁移语言标签", action: "执行迁移", description: "把语言类标签迁移到独立语言字段。", fields: [text("csvPath", "CSV 文件", "data/gallery_info_no_name/JM_info_gender_bender.csv"), text("languageTags", "语言标签", "中文, 英文, 日文"), timeout()] },
      { id: "clean-date", title: "清洗上传日期", action: "清洗日期", description: "统一上传日期格式并修正常见脏数据。", fields: [text("csvPath", "CSV 文件", "data/gallery_info/JM_info_gender_bender_full.csv"), timeout()] },
      { id: "title-words", title: "标题词频统计", action: "生成词频", description: "分词、过滤停用词并输出标题高频词汇。", fields: [text("inputCsv", "输入 CSV", "data/gallery_info/gallery_info_gender_bender_full.csv"), text("outputCsv", "输出 CSV", "data_processing/title_words_frequency.csv"), text("stopWords", "停用词词典", "dictionaries/TITLE_STOP_WORDS.txt"), text("semanticMap", "标题语义映射", "dictionaries/TITLE_SEMANTIC_MAP.json"), timeout(1200)] },
      { id: "tag-set", title: "聚合未映射标签", action: "聚合标签", description: "聚合语义映射中尚未覆盖的标签并导出文本。", fields: [text("csvDir", "CSV 目录", "data/gallery_info"), text("semanticMap", "语义映射 JSON", "dictionaries/SEMANTIC_MAP.json"), text("outputFile", "输出 TXT", "data_processing/aggregated_tags.txt"), timeout()] },
      { id: "map-add-name", title: "语义映射补原名", action: "转换映射", description: "为语义映射条目补充原始名称并另存。", fields: [text("inputFile", "输入 JSON", "data_processing/111.json"), text("outputFile", "输出 JSON", "data_processing/222.json"), timeout()] },
    ],
  },
  {
    id: "database",
    code: "A.2",
    title: "数据库同步",
    scripts: [
      { id: "db-sync", title: "增量同步 CSV 到 MySQL", action: "增量同步", description: "仅同步新增或发生变化的 CSV 记录。", defaultOpen: true, fields: [text("csvDir", "CSV 目录", "data/gallery_info"), timeout(1800, 14400)] },
      { id: "db-rebuild", title: "覆盖重建 MySQL 表", action: "覆盖重建", description: "清空并使用 CSV 全量重建 gallery_info 表。", confirmField: "confirm", fields: [text("csvDir", "CSV 目录", "data/gallery_info"), checkbox("confirm", "确认覆盖 gallery_info 表"), timeout(1800, 14400)] },
      { id: "db-optimize", title: "优化 MySQL 表结构与全文索引", action: "执行优化", description: "执行 ALTER TABLE 与 FULLTEXT 索引维护。", confirmField: "confirm", fields: [checkbox("confirm", "确认执行 ALTER TABLE 与 FULLTEXT 索引维护"), timeout(3600, 14400)] },
    ],
  },
  {
    id: "translation",
    code: "A.3",
    title: "标题 AI 翻译",
    scripts: [
      {
        id: "title-translate",
        title: "标题 AI 翻译",
        action: "开始翻译标题",
        description: "按批次调用兼容 OpenAI 的接口，支持 LM Studio、并发、失败重试与 JSONL 审查流程。",
        defaultOpen: true,
        confirmField: "confirm",
        fields: [
          checkbox("lmStudio", "LM Studio 本地单线程模式", false, "启用后使用本地模型并强制单线程。"),
          text("apiUrl", "调用 URL / Base URL", "https://api.openai.com/v1/chat/completions", "可填完整 /chat/completions 地址或兼容 OpenAI 的 Base URL。"),
          password("apiKey", "API Key", "", "为空时读取环境变量 OPENAI_API_KEY。"),
          text("model", "模型名", "gpt-4o-mini"),
          text("jsonlOutput", "成功 JSONL 输出", "data_processing/title_translation_results.jsonl"),
          text("failedJsonlOutput", "失败 JSONL 输出", "data_processing/title_translation_failed_results.jsonl"),
          checkbox("jsonlOnly", "仅写 JSONL，不回写数据库"),
          number("batchSize", "每组标题数量", 20, 1, 200, 1),
          number("concurrency", "并发组数", 3, 1, 32, 1),
          number("startIndex", "起始序号", 1, 1, 100000000, 1),
          number("endIndex", "结束序号（0 表示不限制）", 0, 0, 100000000, 1),
          number("requestTimeout", "单次请求超时秒数", 120, 10, 3600, 10),
          number("maxRetries", "失败重试次数", 2, 0, 10, 1),
          number("temperature", "Temperature", 0.2, 0, 2, 0.1),
          number("timeout", "整批任务超时秒数", 7200, 10, 86400, 60),
          checkbox("confirm", "确认调用 LLM 并写入标题译文"),
        ],
      },
    ],
  },
  {
    id: "cache",
    code: "A.4",
    title: "缓存与向量",
    scripts: [
      { id: "b64", title: "Base64 预编码", action: "生成 Base64", description: "把线上封面和本地缩略图预编码为文本缓存。", defaultOpen: true, fields: [multiselect("sources", "来源目录", ["线上封面", "本地缩略图"], ["线上封面", "本地缩略图"]), text("cacheDir", "主缓存目录", "b64_cache"), text("tmpDir", "增量输出目录", "b64_tmp"), timeout(1800, 14400)] },
      { id: "text-vector", title: "文本语义向量", action: "构建文本向量", description: "使用本地 embedding 模型生成库存文本向量。", fields: [text("modelPath", "模型目录", "models/Qwen3-Embedding-0.6B"), text("vectorFile", "输出向量文件", "manga_vectors/manga_vectors_Qwen3.pkl"), number("batchSize", "Batch Size", 16, 1, 256, 1), number("maxTextLength", "文本截断长度", 800, 0, 5000, 50), textarea("sql", "SQL", "SELECT * FROM gallery_info WHERE ID != ''"), timeout(7200, 86400)] },
      { id: "clip-vector", title: "封面 CLIP 向量", action: "执行 CLIP 操作", description: "构建、刷新或统计封面向量索引。", fields: [select("action", "操作", "构建/刷新", ["构建/刷新", "统计"]), text("modelPath", "CLIP 模型目录", "models/clip-vit-base-patch32"), text("indexPath", "索引文件", "manga_vectors/clip_image_index.pkl"), text("imageDirs", "图片目录", "onlineimgtmp, localimgtmp"), number("batchSize", "Batch Size", 64, 1, 512, 1), select("device", "设备", "auto", ["auto", "cpu", "cuda"]), checkbox("rebuild", "全量重建"), timeout(7200, 86400)] },
      { id: "cache-delete", title: "缓存清理", action: "删除缓存", description: "删除选中的预处理或向量缓存。", confirmField: "confirm", fields: [multiselect("targets", "清理对象", [], ["预处理 DataFrame", "预处理 Hash", "文本向量", "封面向量"]), checkbox("confirm", "确认删除选中的缓存文件")] },
    ],
  },
  {
    id: "maintenance",
    code: "A.5",
    title: "维护工具",
    scripts: [
      { id: "prefix-rename", title: "图片/缓存 ID 前缀修正", action: "执行重命名", description: "批量修正 Base64 TXT 或本地缩略图文件名中的 ID 前缀。", defaultOpen: true, fields: [select("targetKind", "对象", "Base64 缓存 TXT", ["Base64 缓存 TXT", "本地缩略图"]), text("targetDir", "目录", "b64_cache"), text("prefix", "前缀", "NH"), timeout()] },
      { id: "merge-b64", title: "合并 Base64 增量缓存", action: "合并缓存", description: "把增量缓存合并进主目录，可选择覆盖同名文件。", confirmField: "confirm", fields: [text("tmpDir", "增量目录", "b64_tmp"), text("cacheDir", "主缓存目录", "b64_cache"), checkbox("overwrite", "覆盖同名文件"), checkbox("confirm", "确认合并")] },
      { id: "clean-title-jsonl", title: "清理标题翻译 JSONL failed 条目", action: "清理 failed 条目", description: "移除失败批次记录，并可生成 .bak 备份。", confirmField: "confirm", fields: [text("jsonlPath", "JSONL 文件", "data_processing/title_translation_results.jsonl"), checkbox("keepBackup", "生成 .bak 备份", true), timeout(), checkbox("confirm", "确认清理 failed 条目")] },
      { id: "delete-gallery-rows", title: "按 ID 删除数据库条目", action: "预览/删除数据库条目", description: "从输入或文件读取 ID；未确认时只做预览。", confirmField: "confirm", fields: [textarea("ids", "ID 列表", "", "多个 ID 可用空格、换行或逗号分隔。"), text("idFile", "ID 文件", "", "可选，相对项目根目录。"), timeout(), checkbox("confirm", "确认执行 DELETE 删除整条数据库记录")] },
      { id: "clear-title-translation", title: "按 error.json 清空标题译文", action: "预览/清空标题译文", description: "识别错误文件中的 ID，分批把标题译文清空为 NULL 或空字符串。", confirmField: "confirm", fields: [text("inputFile", "错误 JSON 文件", "tools/error.json"), number("previewLimit", "预览条数", 30, 1, 1000, 1), number("chunkSize", "每批 ID 数量", 500, 1, 5000, 50), checkbox("emptyString", "清空为空字符串（默认清空为 NULL）"), timeout(), checkbox("confirm", "确认清空这些 ID 的标题译文")] },
    ],
  },
];

export const collectionScripts: ScriptDefinition[] = [
  { id: "collection-nh-online", title: "NH 在线抓信息", action: "开始抓取 NH 信息", description: "从 NH 列表页并发采集元数据和缩略图。", confirmField: "confirm", fields: [text("baseUrl", "站点 Base URL", "https://nhentai.net"), text("startUrl", "抓取起始网址", "https://nhentai.net/language/chinese/?sort=date"), number("maxPage", "抓到多少页截止", 1, 1, 100000, 1), text("outputCsv", "保存 CSV 文件名及路径", "gallery_info_chinese.csv"), text("imageDir", "缩略图保存目录", "onlineimgtmp"), text("errorLog", "错误日志路径", "logs/NH_error_log_online.txt"), number("workers", "并发线程数", 10, 1, 64, 1), timeout(1800, 86400), checkbox("confirm", "确认执行采集脚本")] },
  { id: "collection-jm-online", title: "JM 在线抓信息", action: "开始抓取 JM 信息", description: "从 JM 搜索页并发采集元数据和封面。", confirmField: "confirm", fields: [text("baseUrl", "站点 Base URL", "https://18comic.vip"), text("startUrl", "抓取起始网址", "https://18comic.vip/search/photos?search_query=%E7%99%BE%E5%90%88"), number("maxPages", "抓到多少页截止", 80, 1, 100000, 1), text("csvPath", "保存 CSV 文件名及路径", "JM_info_yuri.csv"), text("outputDir", "封面保存目录", "onlineimgtmp"), number("workers", "并发线程数", 5, 1, 64, 1), timeout(3600, 86400), checkbox("confirm", "确认执行采集脚本")] },
  { id: "collection-nh-retry", title: "NH 在线失败页重试", action: "开始 NH 失败页重试", description: "读取 NH 错误日志并重新采集失败列表页。", confirmField: "confirm", fields: [text("baseUrl", "站点 Base URL", "https://nhentai.net"), text("startUrl", "重试页起始网址", "https://nhentai.net/language/chinese/?sort=date"), text("sourceLog", "读取的错误 Log 文件", "logs/NH_error_log_online.txt"), text("retryLog", "输出的错误报告文件", "logs/NH_error_log_online_fix.txt"), text("outputCsv", "保存 CSV 文件名及路径", "gallery_info_chinese.csv"), text("imageDir", "缩略图保存目录", "onlineimgtmp"), number("workers", "并发线程数", 10, 1, 64, 1), timeout(3600, 86400), checkbox("confirm", "确认执行失败页重试")] },
  { id: "collection-jm-retry", title: "JM 在线失败页重试", action: "开始 JM 失败页重试", description: "读取 JM 错误日志并重新采集失败列表页。", confirmField: "confirm", fields: [text("baseUrl", "站点 Base URL", "https://18comic.vip"), text("startUrl", "重试页起始网址", "https://18comic.vip/search/photos?search_query=%E7%99%BE%E5%90%88"), text("sourceLog", "读取的错误 Log 文件", "logs/getjm_errors_20260427_141729.log"), text("retryLog", "输出的错误日志文件", "logs/getjm_fix_retry.log"), text("failedReport", "输出的错误报告 CSV", "logs/failed_pages_retry.csv"), text("csvPath", "保存 CSV 文件名及路径", "JM_info_yuri.csv"), text("outputDir", "封面保存目录", "onlineimgtmp"), number("workers", "并发线程数", 5, 1, 64, 1), timeout(3600, 86400), checkbox("confirm", "确认执行失败页重试")] },
  { id: "collection-nh-local-info", title: "NH 本地链接抓信息", action: "开始解析本地链接", description: "从本地 HTML/TXT 链接列表抓取元数据。", confirmField: "confirm", fields: [text("inputFile", "本地链接 HTML/TXT", "data/local_data/NH_all.txt"), text("outputCsv", "保存 CSV 文件名及路径", "gallery_info_local.csv"), text("errorLog", "错误日志路径", "logs/NH_error_log_local.txt"), number("interval", "请求间隔秒数", 2, 0, 60, 0.5), timeout(1800, 86400), checkbox("confirm", "确认执行采集脚本")] },
  { id: "collection-nh-local-images", title: "NH 本地链接抓图片", action: "开始抓取本地链接图片", description: "从本地链接列表抓取分册图片，并限制单本最大页数。", confirmField: "confirm", fields: [text("inputFile", "本地链接 HTML/TXT", "data/local_data/NH_2.txt"), text("rootDir", "图片保存根目录", "output"), text("errorLog", "错误日志路径", "logs/NH_error_log_images_local.txt"), number("maxPages", "单本最大页数保护", 200, 1, 10000, 1), number("interval", "请求间隔秒数", 1.5, 0, 60, 0.5), number("retries", "页面请求重试次数", 3, 1, 20, 1), timeout(3600, 86400), checkbox("confirm", "确认执行采集脚本")] },
];

export function initialScriptValues(script: ScriptDefinition) {
  return Object.fromEntries(script.fields.map((field) => [field.id, field.defaultValue]));
}
