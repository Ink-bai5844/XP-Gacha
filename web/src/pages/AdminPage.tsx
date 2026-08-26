import { Database, FileArchive, Play, RefreshCw, Square, Terminal, Upload } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import { cancelJob, getJob, getSystemStatus, importBundle, importProject, startJob } from "../api/client";
import {
  collectionScripts,
  dataSections,
  initialScriptValues,
  type ScriptDefinition,
  type ScriptField,
  type ScriptFieldValue,
} from "../data/scripts";
import { useAppState } from "../state/AppState";

type Job = {
  id: string;
  scriptId: string;
  title: string;
  status: "queued" | "running" | "cancelling" | "completed" | "failed" | "cancelled";
  lines: string[];
};

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
    if (script.confirmField && !values[script.confirmField]) {
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

function AppendixTerminal({ job, cancel }: { job: Job | null; cancel: () => void }) {
  return (
    <section className="appendix-terminal" aria-live="polite">
      <header><span><Terminal size={14} />脚本输出</span><b className="mono">{job ? `${job.title} / ${job.status.toUpperCase()}` : "IDLE"}</b>{job && ["queued", "running"].includes(job.status) && <button type="button" onClick={cancel}><Square size={12} />中止任务</button>}</header>
      <pre>{job ? job.lines.join("\n") : "PS D:\\Code\\Python\\XP-Gacha> 等待执行任务……"}</pre>
      {job?.status === "completed" && <p className="terminal-result">任务完成。</p>}
      {job?.status === "cancelled" && <p className="terminal-result">任务已由用户中止。</p>}
      {job?.status === "failed" && <p className="terminal-result">任务执行失败，请检查上方输出。</p>}
    </section>
  );
}

export function AdminPage() {
  const { flash, backendStatus, refreshLibrary } = useAppState();
  const allScripts = useMemo(() => [...dataSections.flatMap((section) => section.scripts), ...collectionScripts], []);
  const [activeSection, setActiveSection] = useState(dataSections[0].id);
  const [collectionMode, setCollectionMode] = useState(collectionScripts[0].id);
  const [statsToken, setStatsToken] = useState(0);
  const [values, setValues] = useState<Record<string, Record<string, ScriptFieldValue>>>(() => Object.fromEntries(allScripts.map((script) => [script.id, initialScriptValues(script)])));
  const [job, setJob] = useState<Job | null>(null);
  const timerRef = useRef<number | null>(null);
  const [system, setSystem] = useState<SystemStatus | null>(null);
  const [importFile, setImportFile] = useState<File | null>(null);
  const [importMode, setImportMode] = useState<"upsert" | "replace">("upsert");
  const [importing, setImporting] = useState(false);
  const [importResult, setImportResult] = useState("");

  useEffect(() => () => {
    if (timerRef.current) window.clearInterval(timerRef.current);
  }, []);

  const loadSystem = () => {
    if (backendStatus !== "online") return;
    void getSystemStatus().then(setSystem).catch((error: Error) => flash(`系统状态读取失败：${error.message}`));
  };

  useEffect(loadSystem, [backendStatus, statsToken]);

  const watchJob = (jobId: string, title: string) => {
    timerRef.current = window.setInterval(() => {
      void getJob(jobId).then((current) => {
        setJob({ id: current.id, scriptId: current.scriptId, title, status: current.status, lines: current.lines });
        if (["completed", "failed", "cancelled"].includes(current.status)) {
          if (timerRef.current) window.clearInterval(timerRef.current);
          timerRef.current = null;
          loadSystem();
          refreshLibrary();
          flash(`${title}：${current.status === "completed" ? "任务完成" : "任务已结束"}`);
        }
      }).catch((error: Error) => {
        if (timerRef.current) window.clearInterval(timerRef.current);
        timerRef.current = null;
        flash(`任务状态读取失败：${error.message}`);
      });
    }, 650);
  };

  const run = (script: ScriptDefinition) => {
    if (job && ["queued", "running", "cancelling"].includes(job.status)) return;
    if (backendStatus !== "online") { flash("后端离线，无法执行真实任务"); return; }
    void startJob(script.id, values[script.id] as Record<string, unknown>).then((started) => {
      setJob({ id: started.id, scriptId: script.id, title: script.title, status: started.status, lines: started.lines });
      watchJob(started.id, script.title);
    }).catch((error: Error) => flash(`启动失败：${error.message}`));
  };

  const cancel = () => {
    if (!job?.id) return;
    void cancelJob(job.id).then((current) => setJob((existing) => existing ? { ...existing, status: current.status, lines: current.lines } : existing)).catch((error: Error) => flash(error.message));
  };

  const runImport = async (projectData = false) => {
    if (backendStatus !== "online") return flash("后端离线，无法导入");
    if (!projectData && !importFile) return flash("请先选择 ZIP 或 CSV 文件");
    if (importMode === "replace" && !window.confirm("覆盖模式会重建 gallery_info 表，确认继续？")) return;
    setImporting(true);
    setImportResult("");
    try {
      const result = projectData ? await importProject(importMode) : await importBundle(importFile!, importMode, true);
      setImportResult(`已导入 ${result.imported} 条，当前总计 ${result.total ?? "—"} 条；识别 CSV ${result.csvFiles} 个。`);
      refreshLibrary();
      loadSystem();
      flash("一键导入完成");
    } catch (error) {
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
        <p>保留原版六个分区、全部脚本参数、确认开关和输出区域。任务现在由后端安全白名单启动，同一时间只允许一个真实进程运行。</p>
      </header>

      <section className="system-overview">
        <div className="overview-toolbar"><span><FileArchive size={14} />一键导入词典 / 数据</span><span className="mono">ZIP 可同时包含 CSV 与四个标准词典文件</span></div>
        <div className="script-fields">
          <label className="admin-field"><span>导入包</span><input type="file" accept=".zip,.csv" onChange={(event) => setImportFile(event.target.files?.[0] ?? null)} /><small>{importFile?.name || "可上传 ZIP 或单个 CSV"}</small></label>
          <label className="admin-field"><span>数据库模式</span><select value={importMode} onChange={(event) => setImportMode(event.target.value as "upsert" | "replace")}><option value="upsert">增量写入 / 更新</option><option value="replace">覆盖重建</option></select></label>
        </div>
        <div className="detail-actions">
          <button type="button" disabled={importing || !importFile} onClick={() => void runImport(false)}><Upload size={14} />{importing ? "导入中…" : "上传并一键导入"}</button>
          <button type="button" disabled={importing} onClick={() => void runImport(true)}><Database size={14} />导入项目 data/gallery_info</button>
        </div>
        {importResult && <p className="terminal-result mono" role="status">{importResult}</p>}
      </section>

      <section className="system-overview">
        <div className="overview-toolbar"><button type="button" onClick={() => { setStatsToken((value) => value + 1); flash("处理统计已刷新"); }}><RefreshCw size={14} />刷新统计</button><span className="mono">大目录文件数默认不自动扫描 · TOKEN {String(statsToken).padStart(3, "0")}</span></div>
        <div className="system-metrics">
          <article><span>CSV</span><strong className="mono">{system?.counts.csv ?? 0}</strong></article>
          <article><span>线上封面</span><strong className="mono">{system?.counts.onlineCovers ?? 0}</strong></article>
          <article><span>本地缩略图</span><strong className="mono">{system?.counts.localThumbnails ?? 0}</strong></article>
          <article><span>Base64</span><strong className="mono">{system?.counts.base64 ?? 0}</strong></article>
          <article><span>数据库</span><strong className="mono">{system?.database.available ? system.database.row_count : "OFF"}</strong></article>
        </div>
        <table className="cache-status-table"><thead><tr><th>项目</th><th>路径</th><th>状态</th><th>大小 KB</th></tr></thead><tbody>
          {system?.caches.map((cache) => <tr key={cache.name}><td>{cache.name}</td><td className="mono">{cache.path}</td><td>{cache.exists ? "存在" : "缺失"}</td><td className="mono number-col">{cache.sizeKb.toLocaleString()}</td></tr>)}
        </tbody></table>
      </section>

      <AppendixTerminal job={job} cancel={cancel} />

      <nav className="appendix-tabs" aria-label="数据处理分区">
        {dataSections.map((section) => <button type="button" className={activeSection === section.id ? "active" : ""} onClick={() => setActiveSection(section.id)} key={section.id}><span className="mono">{section.code}</span>{section.title}</button>)}
        <button type="button" className={activeSection === "collection" ? "active" : ""} onClick={() => setActiveSection("collection")}><span className="mono">A.6</span>采集入口</button>
      </nav>

      {currentSection && (
        <section className="appendix-section">
          <header><span className="mono">{currentSection.code}</span><h3>{currentSection.title}</h3><p>表单参数只保存在当前浏览器会话。</p></header>
          {currentSection.scripts.map((script) => <ScriptPanel script={script} values={values[script.id]} setValues={(next) => setValues((current) => ({ ...current, [script.id]: next }))} activeJob={job} run={run} key={script.id} />)}
        </section>
      )}

      {activeSection === "collection" && (
        <section className="appendix-section">
          <header><span className="mono">A.6</span><h3>采集入口</h3><p>对应原版“流程”选择器，只显示当前采集流程的完整参数。</p></header>
          <label className="collection-mode">流程<select value={collectionMode} onChange={(event) => setCollectionMode(event.target.value)}>{collectionScripts.map((script) => <option value={script.id} key={script.id}>{script.title}</option>)}</select></label>
          <ScriptPanel script={collectionSelected} values={values[collectionSelected.id]} setValues={(next) => setValues((current) => ({ ...current, [collectionSelected.id]: next }))} activeJob={job} run={run} />
        </section>
      )}

      <p className="appendix-warning"><Terminal size={13} />任务在独立子进程中执行；路径受项目目录约束，危险任务仍要求显式确认。</p>
    </div>
  );
}
