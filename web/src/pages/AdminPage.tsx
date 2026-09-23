import { Activity, ArrowLeft, Database, FileArchive, Gauge, Play, RefreshCw, Save, Terminal, Upload } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import { Link, useSearchParams } from "react-router-dom";
import {
  cancelJob,
  getJob,
  getSystemStatus,
  importBundle,
  importProject,
  startJob,
  type JobProgress,
  type JobResponse,
} from "../api/client";
import {
  collectionScripts,
  dataSections,
  initialScriptValues,
  type ScriptDefinition,
  type ScriptField,
  type ScriptFieldValue,
} from "../data/scripts";
import { useAppState } from "../state/AppState";

const TERMINAL_LINE_LIMIT = 500;

type Job = JobResponse & { title: string };

type SystemStatus = Awaited<ReturnType<typeof getSystemStatus>>;

function FieldControl({ field, value, values, onChange }: {
  field: ScriptField;
  value: ScriptFieldValue;
  values: Record<string, ScriptFieldValue>;
  onChange: (value: ScriptFieldValue) => void;
}) {
  const lmDisabled = Boolean(values.lmStudio) && ["apiUrl", "model", "concurrency"].includes(field.id);
  if (field.type === "checkbox") {
    return (
      <label className="admin-check">
        <input type="checkbox" checked={Boolean(value)} onChange={(event) => onChange(event.target.checked)} />
        <span aria-hidden="true" />
        <b>{field.label}</b>
        {field.help && <small>{field.help}</small>}
      </label>
    );
  }
  if (field.type === "multiselect") {
    const selected = Array.isArray(value) ? value : [];
    return (
      <fieldset className="admin-multiselect">
        <legend>{field.label}</legend>
        {field.options?.map((option) => (
          <label key={option}><input type="checkbox" checked={selected.includes(option)} onChange={(event) => onChange(event.target.checked ? [...selected, option] : selected.filter((item) => item !== option))} />{option}</label>
        ))}
      </fieldset>
    );
  }
  return (
    <label className={`admin-field admin-field-${field.type}`}>
      <span>{field.label}</span>
      {field.type === "select" ? (
        <select value={String(value)} onChange={(event) => onChange(event.target.value)} disabled={lmDisabled}>{field.options?.map((option) => <option value={option} key={option}>{option}</option>)}</select>
      ) : field.type === "textarea" ? (
        <textarea rows={4} value={String(value)} onChange={(event) => onChange(event.target.value)} />
      ) : (
        <input
          className={field.type === "number" ? "mono" : ""}
          type={field.type}
          value={String(value)}
          min={field.min}
          max={field.max}
          step={field.step}
          disabled={lmDisabled}
          onChange={(event) => onChange(field.type === "number" ? Number(event.target.value) : event.target.value)}
        />
      )}
      {field.help && <small>{field.help}</small>}
      {lmDisabled && <small>本地单线程模式下由 LM Studio 配置接管。</small>}
    </label>
  );
}

function ScriptPanel({ script, values, setValues, activeJob, run }: {
  script: ScriptDefinition;
  values: Record<string, ScriptFieldValue>;
  setValues: (values: Record<string, ScriptFieldValue>) => void;
  activeJob: Job | null;
  run: (script: ScriptDefinition) => void;
}) {
  const [error, setError] = useState("");
  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (script.confirmField && !values[script.confirmField] && !values.dryRun) {
      setError("请先勾选确认项。");
      return;
    }
    setError("");
    run(script);
  };
  const busy = activeJob ? ["queued", "running", "cancelling"].includes(activeJob.status) : false;
  return (
    <details className="script-panel" open={script.defaultOpen || activeJob?.scriptId === script.id}>
      <summary><span className="mono">{script.id}</span><b>{script.title}</b><small>{script.description}</small></summary>
      <form onSubmit={submit}>
        <div className="script-fields">
          {script.fields.map((field) => (
            <FieldControl
              field={field}
              value={values[field.id]}
              values={values}
              onChange={(value) => setValues({ ...values, [field.id]: value })}
              key={field.id}
            />
          ))}
        </div>
        {error && <p className="form-error" role="alert">{error}</p>}
        <button className="run-script" type="submit" disabled={busy}><Play size={14} />{busy && activeJob?.scriptId !== script.id ? "其他任务正在运行" : script.action}</button>
      </form>
    </details>
  );
}

function finiteNumber(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function formatCount(value: number | null | undefined) {
  const number = finiteNumber(value);
  return number === null ? "—" : Math.max(0, Math.round(number)).toLocaleString("zh-CN");
}

function formatTimestamp(value: JobProgress["updatedAt"]) {
  if (value === null || value === undefined || value === "") return "—";
  const timestamp = typeof value === "number" && value < 1_000_000_000_000 ? value * 1000 : value;
  const date = new Date(timestamp);
  return Number.isNaN(date.getTime())
    ? String(value)
    : date.toLocaleTimeString("zh-CN", { hour12: false });
}

function progressStage(stage: string | null | undefined) {
  if (!stage) return "等待采集器状态";
  const labels: Record<string, string> = {
    initializing: "初始化",
    ready: "初始化完成",
    round_start: "开始本轮采集",
    discovery: "发现列表页",
    discovering: "发现列表页",
    streaming: "流式发现与采集",
    collecting: "采集详情与图片",
    processing: "采集详情与图片",
    downloading: "下载分册图片",
    retry: "重试缺失项",
    retrying: "重试缺失项",
    retry_wait: "等待下一轮重试",
    saving: "写入 CSV 与断点",
    stopping: "安全保存并停止",
    stopped: "已安全停止",
    completed: "采集完成",
    interrupted: "已安全停止",
    failed: "采集失败",
  };
  return labels[stage.toLowerCase()] ?? stage;
}

function capTerminalLines(lines: string[]) {
  return lines.length > TERMINAL_LINE_LIMIT ? lines.slice(-TERMINAL_LINE_LIMIT) : lines;
}

function CollectionProgressPanel({ job }: { job: Job | null }) {
  if (!job?.scriptId.startsWith("collection-")) return null;
  const progress = job.progress ?? {};
  const hasProgress = Object.values(progress).some((value) => value !== null && value !== undefined);
  const rawPercent = finiteNumber(progress?.progressPercent);
  const percent = rawPercent === null ? null : Math.min(100, Math.max(0, rawPercent));
  const lastWindowSuccess = finiteNumber(progress?.lastWindowSuccess);
  const previousWindowSuccess = finiteNumber(progress?.previousWindowSuccess);
  const windowElapsed = finiteNumber(progress?.windowElapsedSeconds);
  const concurrency = finiteNumber(progress?.currentConcurrency);
  const maxConcurrency = finiteNumber(progress?.maxConcurrency);
  const active = ["queued", "running", "cancelling"].includes(job.status);
  const trendLabels: Record<string, string> = {
    warming_up: "启动预热中（首个完整窗口不计分）",
    stabilizing: "正在采样稳定吞吐",
    probing_down: "正在测试更低并发的成功吞吐",
    probing_up: "正在测试更高并发的成功吞吐",
    holding: "保持当前高效档位继续观察",
  };
  const throughputTrend = progress?.throughputTrend
    ? (trendLabels[progress.throughputTrend] ?? progress.throughputTrend)
    : "等待吞吐量窗口样本";
  const throughputLevel = lastWindowSuccess === null || previousWindowSuccess === null
    ? "unknown"
    : lastWindowSuccess > previousWindowSuccess
      ? "low"
      : lastWindowSuccess < previousWindowSuccess
        ? "high"
        : "stable";
  const metrics = [
    ["完全成功", progress?.complete, "complete"],
    ["缺信息", progress?.missingInfo, "missing"],
    ["缺图片", progress?.missingImage, "missing"],
    ["信息和图片全缺", progress?.missingBoth, "missing"],
    ["待处理", progress?.pending, "pending"],
    ["不可重试", progress?.terminal, "terminal"],
  ] as const;

  return (
    <section className="collection-progress" aria-live="polite" aria-busy={active && !hasProgress}>
      <header className="collection-progress-heading">
        <span><Activity size={15} />实时采集进度</span>
        <div>
          <b>{progressStage(progress?.stage)}</b>
          <small className="mono">{progress?.mode || job.scriptId.replace("collection-", "")} · 第 {formatCount(progress?.round)} 轮</small>
        </div>
        <strong className="mono">{percent === null ? "—" : `${percent.toFixed(1)}%`}</strong>
      </header>

      <div className="collection-progress-track" role="progressbar" aria-label="采集总进度" aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent ?? undefined}>
        <span style={{ width: `${percent ?? 0}%` }} />
      </div>

      {!hasProgress ? (
        <p className="collection-progress-empty">任务已创建，正在等待采集器上报第一份进度数据……</p>
      ) : (
        <>
          <div className="collection-progress-metrics">
            {metrics.map(([label, value, tone]) => (
              <article className={`collection-progress-metric collection-progress-metric-${tone}`} key={label}>
                <span>{label}</span><strong className="mono">{formatCount(value)}</strong>
              </article>
            ))}
          </div>

          <div className="collection-progress-details">
            <article>
              <span>列表页</span>
              <strong className="mono">{formatCount(progress.pagesCompleted)} / {formatCount(progress.pagesTotal)}</strong>
              <small className="mono">剩余 {formatCount(progress.pagesPending)} 页</small>
            </article>
            <article>
              <span>已发现项目</span>
              <strong className="mono">{formatCount(progress.discovered)}</strong>
              <small className="mono">CSV 实时写入 {formatCount(progress.csvWrites)} 次</small>
            </article>
            <article>
              <span><Gauge size={13} />动态并发</span>
              <strong className="mono">{formatCount(progress.currentConcurrency)} / {formatCount(progress.maxConcurrency)}</strong>
              <small>当前 / 输入上限</small>
            </article>
            <article className={`collection-window-health collection-window-health-${throughputLevel}`}>
              <span>最近 30 秒成功吞吐</span>
              <strong className="mono">当前 {formatCount(progress.windowSuccess)} · 上轮 {formatCount(progress.lastWindowSuccess)}</strong>
              <small>前轮 {formatCount(progress.previousWindowSuccess)} · 窗口 {windowElapsed === null ? "—" : `${Math.min(30, Math.max(0, windowElapsed)).toFixed(1)} / 30 秒`} · {throughputTrend}</small>
            </article>
          </div>
        </>
      )}

      <footer className="collection-progress-footer mono">
        <span>状态：{job.status.toUpperCase()}</span>
        <span>最近更新：{formatTimestamp(progress?.updatedAt)}</span>
      </footer>
    </section>
  );
}

function AppendixTerminal({ job, cancel }: { job: Job | null; cancel: () => void }) {
  const truncatedLines = job ? Math.max(0, job.lineCount - job.lines.length) : 0;
  const canCancel = job ? ["queued", "running", "cancelling"].includes(job.status) : false;
  const cancelling = job?.status === "cancelling";
  const collectionJob = job?.scriptId.startsWith("collection-") ?? false;
  return (
    <section className="appendix-terminal" aria-live="polite">
      <header><span><Terminal size={14} />脚本输出</span><b className="mono">{job ? `${job.title} / ${job.status.toUpperCase()}` : "IDLE"}</b>{canCancel && <button type="button" onClick={cancel} disabled={cancelling} aria-busy={cancelling}><Save size={12} />{cancelling ? (collectionJob ? "正在安全保存并停止…" : "正在中止…") : (collectionJob ? "安全保存并停止" : "中止任务")}</button>}</header>
      {truncatedLines > 0 && <p className="terminal-truncation mono">前 {truncatedLines.toLocaleString("zh-CN")} 行已截断，仅显示最新 {TERMINAL_LINE_LIMIT} 行（服务端累计 {job?.lineCount.toLocaleString("zh-CN")} 行）。</p>}
      <pre>{job ? job.lines.join("\n") : "PS D:\\Code\\Python\\XP-Gacha> 等待执行任务……"}</pre>
      {job?.status === "completed" && <p className="terminal-result">任务完成。</p>}
      {job?.status === "cancelling" && <p className="terminal-result">{collectionJob ? "正在写回 CSV 与断点并等待采集器退出，请勿关闭窗口。" : "正在中止任务……"}</p>}
      {job?.status === "cancelled" && <p className="terminal-result">{collectionJob ? "任务已安全保存并停止。" : "任务已中止。"}</p>}
      {job?.status === "failed" && <p className="terminal-result">任务执行失败，请检查上方输出。</p>}
    </section>
  );
}

export function AdminPage() {
  const { flash, backendStatus, refreshLibrary } = useAppState();
  const [searchParams, setSearchParams] = useSearchParams();
  const requestedSection = searchParams.get("section");
  const initialSection = dataSections.some((section) => section.id === requestedSection) || requestedSection === "collection"
    ? requestedSection!
    : dataSections[0].id;
  const allScripts = useMemo(() => [...dataSections.flatMap((section) => section.scripts), ...collectionScripts], []);
  const [activeSection, setActiveSection] = useState(initialSection);
  const [collectionMode, setCollectionMode] = useState(collectionScripts[0].id);
  const [values, setValues] = useState<Record<string, Record<string, ScriptFieldValue>>>(() => Object.fromEntries(allScripts.map((script) => [script.id, initialScriptValues(script)])));
  const [job, setJob] = useState<Job | null>(null);
  const timerRef = useRef<number | null>(null);
  const jobCursorRef = useRef(0);
  const jobWatchTokenRef = useRef(0);
  const importSectionRef = useRef<HTMLElement | null>(null);
  const importFileInputRef = useRef<HTMLInputElement | null>(null);
  const statsRequestRef = useRef(0);
  const [system, setSystem] = useState<SystemStatus | null>(null);
  const [statsLoading, setStatsLoading] = useState(false);
  const [statsRefreshing, setStatsRefreshing] = useState(false);
  const [statsError, setStatsError] = useState("");
  const [importFile, setImportFile] = useState<File | null>(null);
  const [importMode, setImportMode] = useState<"upsert" | "replace">("upsert");
  const [importing, setImporting] = useState(false);
  const [importResult, setImportResult] = useState("");
  const [importSucceeded, setImportSucceeded] = useState(false);

  useEffect(() => () => {
    jobWatchTokenRef.current += 1;
    if (timerRef.current) window.clearTimeout(timerRef.current);
  }, []);

  useEffect(() => {
    if (!requestedSection) return;
    const valid = dataSections.some((section) => section.id === requestedSection) || requestedSection === "collection";
    if (!valid) return;
    setActiveSection(requestedSection);
    window.requestAnimationFrame(() => {
      document.getElementById("appendix-workbench")?.scrollIntoView({ block: "start" });
    });
  }, [requestedSection]);

  const importFocusRequested = searchParams.get("focus") === "import";
  useEffect(() => {
    if (!importFocusRequested) return;
    const frame = window.requestAnimationFrame(() => {
      importSectionRef.current?.scrollIntoView({ block: "start" });
      importFileInputRef.current?.focus({ preventScroll: true });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [importFocusRequested]);

  const selectSection = (sectionId: string) => {
    setActiveSection(sectionId);
    setSearchParams({ section: sectionId }, { replace: true });
  };

  const loadSystem = useCallback(async (refresh = false) => {
    if (backendStatus !== "online") return false;
    const requestId = statsRequestRef.current + 1;
    statsRequestRef.current = requestId;
    setStatsLoading(true);
    setStatsRefreshing(refresh);
    setStatsError("");
    try {
      const payload = await getSystemStatus({ refresh });
      if (statsRequestRef.current === requestId) setSystem(payload);
      return true;
    } catch (error) {
      if (statsRequestRef.current === requestId) {
        const message = (error as Error).message;
        setStatsError(message);
        flash(`系统状态读取失败：${message}`);
      }
      return false;
    } finally {
      if (statsRequestRef.current === requestId) {
        setStatsLoading(false);
        setStatsRefreshing(false);
      }
    }
  }, [backendStatus, flash]);

  useEffect(() => {
    void loadSystem(false);
  }, [loadSystem]);

  const refreshStats = async () => {
    if (await loadSystem(true)) flash("处理统计已重新扫描");
  };

  const watchJob = (jobId: string, title: string) => {
    jobWatchTokenRef.current += 1;
    const watchToken = jobWatchTokenRef.current;
    if (timerRef.current) window.clearTimeout(timerRef.current);

    const poll = async () => {
      if (watchToken !== jobWatchTokenRef.current) return;
      let finished = false;
      try {
        const current = await getJob(jobId, jobCursorRef.current);
        if (watchToken !== jobWatchTokenRef.current) return;
        jobCursorRef.current = current.lineCount;
        setJob((existing) => ({
          ...current,
          title,
          lines: capTerminalLines(existing?.id === current.id ? [...existing.lines, ...current.lines] : current.lines),
          progress: current.progress ?? (existing?.id === current.id ? existing.progress : null),
        }));
        if (["completed", "failed", "cancelled"].includes(current.status)) {
          finished = true;
          void loadSystem();
          refreshLibrary();
          flash(`${title}：${current.status === "completed" ? "任务完成" : "任务已结束"}`);
        }
      } catch (error) {
        if (watchToken !== jobWatchTokenRef.current) return;
        finished = true;
        flash(`任务状态读取失败：${(error as Error).message}`);
      } finally {
        if (watchToken === jobWatchTokenRef.current && !finished) {
          timerRef.current = window.setTimeout(() => void poll(), 650);
        } else if (watchToken === jobWatchTokenRef.current) {
          timerRef.current = null;
        }
      }
    };

    timerRef.current = window.setTimeout(() => void poll(), 650);
  };

  const run = (script: ScriptDefinition) => {
    if (job && ["queued", "running", "cancelling"].includes(job.status)) return;
    if (backendStatus !== "online") { flash("后端离线，无法执行真实任务"); return; }
    void startJob(script.id, values[script.id] as Record<string, unknown>).then((started) => {
      jobCursorRef.current = started.lineCount;
      setJob({ ...started, title: script.title, lines: capTerminalLines(started.lines) });
      watchJob(started.id, script.title);
    }).catch((error: Error) => flash(`启动失败：${error.message}`));
  };

  const cancel = () => {
    if (!job?.id) return;
    const cancellingJob = job;
    jobWatchTokenRef.current += 1;
    if (timerRef.current) window.clearTimeout(timerRef.current);
    timerRef.current = null;
    setJob((existing) => existing?.id === cancellingJob.id ? { ...existing, status: "cancelling" } : existing);
    void cancelJob(cancellingJob.id, jobCursorRef.current).then((current) => {
      jobCursorRef.current = current.lineCount;
      setJob((existing) => existing ? {
        ...current,
        title: existing.title,
        lines: capTerminalLines(existing.id === current.id ? [...existing.lines, ...current.lines] : current.lines),
        progress: current.progress ?? existing.progress,
      } : existing);
      if (["queued", "running", "cancelling"].includes(current.status)) watchJob(current.id, cancellingJob.title);
    }).catch((error: Error) => {
      setJob((existing) => existing?.id === cancellingJob.id ? { ...existing, status: cancellingJob.status } : existing);
      flash(error.message);
      watchJob(cancellingJob.id, cancellingJob.title);
    });
  };

  const runImport = async (projectData = false) => {
    if (backendStatus !== "online") return flash("后端离线，无法导入");
    if (!projectData && !importFile) return flash("请先选择 ZIP 或 CSV 文件");
    if (importMode === "replace" && !window.confirm("覆盖模式会重建 gallery_info 表，确认继续？")) return;
    setImporting(true);
    setImportResult("");
    setImportSucceeded(false);
    try {
      const result = projectData ? await importProject(importMode) : await importBundle(importFile!, importMode, true);
      setImportResult(`已导入 ${result.imported} 条，当前总计 ${result.total ?? "—"} 条；识别 CSV ${result.csvFiles} 个。`);
      setImportSucceeded((result.total ?? 0) > 0);
      refreshLibrary();
      void loadSystem();
      flash("一键导入完成");
    } catch (error) {
      setImportSucceeded(false);
      setImportResult(`导入失败：${(error as Error).message}`);
    } finally {
      setImporting(false);
    }
  };

  const currentSection = dataSections.find((section) => section.id === activeSection);
  const collectionSelected = collectionScripts.find((script) => script.id === collectionMode) ?? collectionScripts[0];

  return (
    <div className="admin-page">
      <header className="page-intro page-intro-split">
        <div><span className="section-code">APPENDIX A / PROCESSING</span><h2>附录 A<br />数据处理</h2></div>
        <p>用于导入馆藏数据、维护词典与缓存、执行采集和索引任务；同一时间只运行一个后台任务。</p>
      </header>

      <section id="one-click-import" ref={importSectionRef} className="system-overview import-overview" aria-busy={importing}>
        <div className="overview-toolbar"><span><FileArchive size={14} />一键导入词典 / 数据</span><span className="mono">ZIP 可同时包含 CSV 与四个标准词典文件</span></div>
        <div className="script-fields">
          <label className="admin-field"><span>导入包</span><input ref={importFileInputRef} type="file" accept=".zip,.csv" aria-describedby="import-file-help" onChange={(event) => setImportFile(event.target.files?.[0] ?? null)} /><small id="import-file-help">{importFile?.name || "可上传 ZIP 或单个 CSV；input_data.zip 无需解压"}</small></label>
          <label className="admin-field"><span>数据库模式</span><select value={importMode} onChange={(event) => setImportMode(event.target.value as "upsert" | "replace")}><option value="upsert">增量写入 / 更新</option><option value="replace">覆盖重建</option></select></label>
        </div>
        <div className="detail-actions">
          <button type="button" disabled={importing || !importFile} onClick={() => void runImport(false)}><Upload size={14} />{importing ? "导入中…" : "上传并一键导入"}</button>
          <button type="button" disabled={importing} onClick={() => void runImport(true)}><Database size={14} />导入项目 data/gallery_info</button>
        </div>
        {importResult && <p className="terminal-result mono" role={importResult.startsWith("导入失败：") ? "alert" : "status"}>{importResult}</p>}
        {importSucceeded && <Link className="import-return-link" to="/"><ArrowLeft size={14} />返回库存目录查看已导入数据</Link>}
      </section>

      <section className="system-overview" aria-busy={statsLoading}>
        <div className="overview-toolbar">
          <button type="button" disabled={backendStatus !== "online" || statsLoading} onClick={() => void refreshStats()}>
            <RefreshCw size={14} />{statsRefreshing ? "正在重新统计…" : statsLoading ? "正在读取…" : "刷新统计"}
          </button>
          <span className="mono">
            {statsRefreshing
              ? "正在重新扫描大目录，请稍候"
              : statsLoading
                ? "正在读取已缓存的统计"
                : statsError
                  ? `读取失败：${statsError}`
                  : system
                    ? "普通加载使用缓存 · 点击按钮才重新扫描"
                    : backendStatus === "offline" ? "后端离线" : "等待后端连接"}
          </span>
        </div>
        <div className="system-metrics">
          <article><span>CSV</span><strong className="mono">{system ? system.counts.csv : statsLoading ? "…" : "—"}</strong></article>
          <article><span>线上封面</span><strong className="mono">{system ? system.counts.onlineCovers : statsLoading ? "…" : "—"}</strong></article>
          <article><span>本地缩略图</span><strong className="mono">{system ? system.counts.localThumbnails : statsLoading ? "…" : "—"}</strong></article>
          <article><span>Base64</span><strong className="mono">{system ? system.counts.base64 : statsLoading ? "…" : "—"}</strong></article>
          <article><span>数据库</span><strong className="mono">{system ? system.database.available ? system.database.row_count : "OFF" : statsLoading ? "…" : "—"}</strong></article>
        </div>
        <table className="cache-status-table"><thead><tr><th>项目</th><th>路径</th><th>状态</th><th>大小 KB</th></tr></thead><tbody>
          {system
            ? system.caches.map((cache) => <tr key={cache.name}><td>{cache.name}</td><td className="mono">{cache.path}</td><td>{cache.exists ? "存在" : "缺失"}</td><td className="mono number-col">{cache.sizeKb.toLocaleString()}</td></tr>)
            : <tr><td colSpan={4}>{statsLoading ? "正在读取统计缓存…" : "暂无统计数据"}</td></tr>}
        </tbody></table>
      </section>

      <CollectionProgressPanel job={job} />
      <AppendixTerminal job={job} cancel={cancel} />

      <nav id="appendix-workbench" className="appendix-tabs" aria-label="数据处理分区">
        {dataSections.map((section) => <button type="button" className={activeSection === section.id ? "active" : ""} onClick={() => selectSection(section.id)} key={section.id}><span className="mono">{section.code}</span>{section.title}</button>)}
        <button type="button" className={activeSection === "collection" ? "active" : ""} onClick={() => selectSection("collection")}><span className="mono">A.6</span>采集入口</button>
      </nav>

      {currentSection && (
        <section className="appendix-section">
          <header><span className="mono">{currentSection.code}</span><h3>{currentSection.title}</h3><p>表单参数只保存在当前浏览器会话。</p></header>
          {currentSection.scripts.map((script) => <ScriptPanel script={script} values={values[script.id]} setValues={(next) => setValues((current) => ({ ...current, [script.id]: next }))} activeJob={job} run={run} key={script.id} />)}
        </section>
      )}

      {activeSection === "collection" && (
        <section className="appendix-section">
          <header><span className="mono">A.6</span><h3>采集入口</h3><p>选择采集流程，并配置范围、输出路径、并发数与自动重试参数。</p></header>
          <label className="collection-mode">流程<select value={collectionMode} onChange={(event) => setCollectionMode(event.target.value)}>{collectionScripts.map((script) => <option value={script.id} key={script.id}>{script.title}</option>)}</select></label>
          <ScriptPanel script={collectionSelected} values={values[collectionSelected.id]} setValues={(next) => setValues((current) => ({ ...current, [collectionSelected.id]: next }))} activeJob={job} run={run} />
        </section>
      )}

      <p className="appendix-warning"><Terminal size={13} />任务在独立子进程中执行；路径受项目目录约束，危险任务仍要求显式确认。</p>
    </div>
  );
}
