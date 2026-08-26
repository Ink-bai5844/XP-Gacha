import { AlertTriangle, BrainCircuit, ExternalLink, Image, X } from "lucide-react";
import { useEffect, useId, useRef } from "react";
import { Link } from "react-router-dom";
import type { SearchCapabilityStatus, SearchDependencyStatus, SystemStatus } from "../api/client";

export type SearchCapabilityKind = "semantic" | "cover";

type SearchUnavailableDialogProps = {
  open: boolean;
  kinds: SearchCapabilityKind[];
  errors: string[];
  status: SystemStatus | null;
  loading: boolean;
  statusError: string;
  onClose: () => void;
};

const capabilityLabels: Record<SearchCapabilityKind, string> = {
  semantic: "AI 语义检索",
  cover: "封面相似检索",
};

function dependencyStateLabel(dependency: SearchDependencyStatus) {
  if (dependency.ready) return "已就绪";
  return dependency.state === "incomplete" ? "目录或文件不完整" : "缺失";
}

function missingLabel(capability: SearchCapabilityStatus | undefined, statusError: string) {
  if (statusError) return "状态检查失败";
  if (!capability) return "正在检查缺失项";
  if (!capability.missing.length) return "文件存在，但运行时加载失败";
  if (capability.missing.length === 2) return "模型和预处理向量全部缺失";
  return capability.missing[0] === "model" ? "缺少模型" : "缺少预处理向量";
}

function DependencyRow({ dependency }: { dependency: SearchDependencyStatus }) {
  return (
    <div className={`search-dependency search-dependency-${dependency.ready ? "ready" : "missing"}`}>
      <div>
        <span>{dependency.kind === "model" ? "本地模型" : "预处理向量 / 索引"}</span>
        <strong>{dependency.label}</strong>
      </div>
      <b>{dependencyStateLabel(dependency)}</b>
      <code>{dependency.path}</code>
      {dependency.kind === "model" ? (
        <a href={dependency.downloadUrl} target="_blank" rel="noreferrer">
          打开官方模型仓库 <ExternalLink size={13} aria-hidden="true" />
        </a>
      ) : (
        <small>该文件与当前库存数据相关，不能下载通用版本，需要在本项目中生成。</small>
      )}
    </div>
  );
}

function CapabilityCard({ kind, capability, statusError }: {
  kind: SearchCapabilityKind;
  capability: SearchCapabilityStatus | undefined;
  statusError: string;
}) {
  const Icon = kind === "semantic" ? BrainCircuit : Image;
  const model = capability?.dependencies.model;
  const vector = capability?.dependencies.vector;
  return (
    <article className="search-capability-card">
      <header>
        <Icon size={20} strokeWidth={1.5} aria-hidden="true" />
        <div>
          <span className="mono">{kind === "semantic" ? "SEMANTIC" : "COVER / CLIP"}</span>
          <h3>{capability?.label ?? capabilityLabels[kind]}</h3>
        </div>
        <strong>{missingLabel(capability, statusError)}</strong>
      </header>

      {model && vector ? (
        <div className="search-dependency-list">
          <DependencyRow dependency={model} />
          <DependencyRow dependency={vector} />
        </div>
      ) : (
        <p className="search-diagnostic-loading mono">
          {statusError ? `状态检查失败：${statusError}` : "正在读取模型与向量状态…"}
        </p>
      )}

      <div className="search-setup-steps">
        <h4>恢复步骤</h4>
        {kind === "semantic" ? (
          <ol>
            <li>从上面的官方链接下载完整的 <code>Qwen/Qwen3-Embedding-0.6B</code> 仓库；配置、Tokenizer、Pooling 与权重文件必须全部保留。</li>
            <li>把整个仓库目录放到“本地模型”显示的目标路径。</li>
            <li>进入“附录 → A.4 缓存与向量 → 文本语义向量”，确认模型目录和输出向量路径与上方一致，然后点击“构建文本向量”。</li>
            <li>任务完成后返回库存重试；如果刚替换过已经加载的旧模型或向量，请重启应用再试。</li>
          </ol>
        ) : (
          <ol>
            <li>从上面的官方链接下载完整的 <code>openai/clip-vit-base-patch32</code> 仓库，并保留 PyTorch 权重、Processor、Tokenizer 与配置文件。</li>
            <li>把模型目录放到“本地模型”显示的目标路径，并确保封面图片已经位于 <code>onlineimgtmp</code> 或 <code>localimgtmp</code>。</li>
            <li>进入“附录 → A.4 缓存与向量 → 封面 CLIP 向量”，选择“构建/刷新”，核对模型和索引路径后执行。</li>
            <li>库内 ID 检索只依赖封面索引；上传图片检索同时依赖 CLIP 模型和封面索引。任务完成后重试，替换旧文件后建议重启应用。</li>
          </ol>
        )}
      </div>
    </article>
  );
}

export function SearchUnavailableDialog({
  open,
  kinds,
  errors,
  status,
  loading,
  statusError,
  onClose,
}: SearchUnavailableDialogProps) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const titleId = useId();
  const descriptionId = useId();

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    if (open && !dialog.open) dialog.showModal();
    if (!open && dialog.open) dialog.close();
  }, [open]);

  return (
    <dialog
      ref={dialogRef}
      className="search-unavailable-dialog"
      aria-labelledby={titleId}
      aria-describedby={descriptionId}
      aria-busy={loading}
      onCancel={(event) => { event.preventDefault(); onClose(); }}
      onClose={() => { if (open) onClose(); }}
      onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}
    >
      <div className="search-diagnostic-sheet">
        <header className="search-diagnostic-heading">
          <div className="search-diagnostic-mark" aria-hidden="true"><AlertTriangle size={22} strokeWidth={1.5} /></div>
          <div>
            <span className="section-code">SEARCH CAPABILITY / SETUP</span>
            <h2 id={titleId}>检索能力尚未就绪</h2>
            <p id={descriptionId}>已检查当前运行环境。请补齐下列模型和数据专属向量，再重新执行检索。</p>
          </div>
          <button type="button" className="search-dialog-close" onClick={onClose} aria-label="关闭检索能力说明">
            <X size={18} aria-hidden="true" />
          </button>
        </header>

        <div className="search-capability-grid">
          {kinds.map((kind) => (
            <CapabilityCard key={kind} kind={kind} capability={status?.searchCapabilities?.[kind]} statusError={statusError} />
          ))}
        </div>

        <section className="search-placement-note" aria-label="不同运行方式的放置路径">
          <h3>文件应该放在哪里？</h3>
          <ul>
            <li><strong>源码 / 便携版：</strong>放在项目或发行包根目录的 <code>models/…</code> 与 <code>manga_vectors/…</code>，执行 A.4 任务时直接使用相同的相对路径。</li>
            <li><strong>Docker：</strong>宿主项目目录会映射到弹窗显示的 <code>/app/…</code> 路径。</li>
          </ul>
        </section>

        {(errors.length > 0 || statusError) && (
          <details className="search-raw-errors">
            <summary>查看本次原始错误</summary>
            {errors.map((error) => <code key={error}>{error}</code>)}
            {statusError && <code>状态检查失败：{statusError}</code>}
          </details>
        )}

        <footer className="search-diagnostic-actions">
          <Link to="/admin?section=cache#appendix-workbench" viewTransition onClick={onClose}>前往附录 A.4</Link>
          <button type="button" onClick={onClose}>知道了</button>
        </footer>
      </div>
    </dialog>
  );
}
