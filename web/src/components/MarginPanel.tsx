import {
  AlertTriangle,
  BrainCircuit,
  ChevronLeft,
  ChevronRight,
  FileImage,
  Image,
  Search,
} from "lucide-react";
import { memo, useCallback, useDeferredValue, useEffect, useId, useMemo, useRef, useState, type CSSProperties, type ReactNode } from "react";
import { getSystemStatus, searchCoverFile, searchMetaOptions, type OptionKind, type OptionSearchResponse, type SystemStatus } from "../api/client";
import { SearchUnavailableDialog, type SearchCapabilityKind } from "./SearchUnavailableDialog";
import { titleWordOptions, useAppState, type WeightState } from "../state/AppState";

type MarginPanelProps = { open: boolean; onToggle: () => void };

const weightLabels: Record<keyof WeightState, string> = {
  tag: "标签总分倍率",
  artist: "作者总分倍率",
  title: "标题总分倍率",
  history: "历史偏好总分倍率",
};

const MAX_RENDERED_OPTIONS = 80;
const MAX_SELECTED_CHIPS = 16;
const INPUT_COMMIT_DELAY = 180;

function BufferedRange({ label, value, min, max, step, suffix = "", onCommit }: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  suffix?: string;
  onCommit: (value: number) => void;
}) {
  const [draft, setDraft] = useState(value);
  const draftRef = useRef(value);
  const draggingRef = useRef(false);
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    draftRef.current = value;
    setDraft(value);
  }, [value]);

  useEffect(() => () => {
    if (timerRef.current !== null) window.clearTimeout(timerRef.current);
  }, []);

  const commit = (next = draftRef.current) => {
    if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    timerRef.current = null;
    if (next !== value) onCommit(next);
  };

  const queueCommit = (next: number) => {
    draftRef.current = next;
    setDraft(next);
    if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    if (!draggingRef.current) {
      timerRef.current = window.setTimeout(() => commit(next), INPUT_COMMIT_DELAY);
    }
  };

  const precision = String(step).includes(".") ? String(step).split(".")[1].length : 0;
  const formatted = precision ? draft.toFixed(precision) : String(draft);
  const position = max === min ? 0 : ((draft - min) / (max - min)) * 100;

  return (
    <label className="editorial-range">
      <span>{label}</span>
      <output className="mono" aria-label={`${label} ${formatted}`}>{formatted}{suffix}</output>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={draft}
        aria-label={label}
        aria-valuetext={`${formatted}${suffix}`}
        onPointerDown={() => { draggingRef.current = true; }}
        onPointerUp={() => { draggingRef.current = false; commit(); }}
        onPointerCancel={() => { draggingRef.current = false; commit(); }}
        onKeyUp={() => commit()}
        onBlur={() => { draggingRef.current = false; commit(); }}
        onChange={(event) => queueCommit(Number(event.target.value))}
        style={{ "--position": `${position}%` } as CSSProperties}
      />
    </label>
  );
}

function BufferedNumber({ label, value, min, max, step, onCommit }: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onCommit: (value: number) => void;
}) {
  const [draft, setDraft] = useState(String(value));
  const nextRef = useRef<number | null>(value);
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    setDraft(String(value));
    nextRef.current = value;
  }, [value]);

  useEffect(() => () => {
    if (timerRef.current !== null) window.clearTimeout(timerRef.current);
  }, []);

  const commit = () => {
    if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    timerRef.current = null;
    if (nextRef.current !== null && nextRef.current !== value) onCommit(nextRef.current);
  };

  return (
    <input
      className="mono"
      aria-label={label}
      type="number"
      min={min}
      max={max}
      step={step}
      value={draft}
      onBlur={commit}
      onKeyDown={(event) => { if (event.key === "Enter") event.currentTarget.blur(); }}
      onChange={(event) => {
        const nextDraft = event.target.value;
        const parsed = Number(nextDraft);
        setDraft(nextDraft);
        nextRef.current = nextDraft === "" || !Number.isFinite(parsed) ? null : Math.min(max, Math.max(min, parsed));
        if (timerRef.current !== null) window.clearTimeout(timerRef.current);
        if (nextRef.current !== null) timerRef.current = window.setTimeout(commit, INPUT_COMMIT_DELAY);
      }}
    />
  );
}

function MultiSelector({ title, label, options, optionKind, optionTotal, values, onChange, defaultOpen = false, children }: {
  title: string;
  label: string;
  options: string[];
  optionKind?: OptionKind;
  optionTotal?: number;
  values: string[];
  onChange: (values: string[]) => void;
  defaultOpen?: boolean;
  children?: ReactNode;
}) {
  const searchId = useId();
  const [open, setOpen] = useState(defaultOpen);
  const [query, setQuery] = useState("");
  const [offset, setOffset] = useState(0);
  const [remoteResult, setRemoteResult] = useState<OptionSearchResponse | null>(null);
  const [loadingOptions, setLoadingOptions] = useState(false);
  const deferredQuery = useDeferredValue(query);
  const normalizedQuery = deferredQuery.trim().toLowerCase();
  const selectedSet = useMemo(() => new Set(values), [values]);

  useEffect(() => {
    if (!open || !optionKind) {
      setRemoteResult(null);
      setLoadingOptions(false);
      return;
    }
    const controller = new AbortController();
    let active = true;
    setLoadingOptions(true);
    const timer = window.setTimeout(() => {
      void searchMetaOptions(optionKind, deferredQuery.trim(), MAX_RENDERED_OPTIONS, offset, controller.signal)
        .then((payload) => { if (active) setRemoteResult(payload); })
        .catch((error: Error) => {
          if (active && error.name !== "AbortError") setRemoteResult(null);
        })
        .finally(() => { if (active) setLoadingOptions(false); });
    }, normalizedQuery ? 140 : 0);
    return () => {
      active = false;
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [deferredQuery, normalizedQuery, offset, open, optionKind]);

  const sourceOptions = remoteResult?.items ?? options;
  const totalOptions = optionTotal || options.length;
  const optionResult = useMemo(() => {
    if (!open) return { visible: [] as string[], total: totalOptions, start: 0, hasPrevious: false, hasMore: false };

    if (optionKind && remoteResult) {
      return {
        visible: remoteResult.items,
        total: remoteResult.total,
        start: remoteResult.offset,
        hasPrevious: remoteResult.offset > 0,
        hasMore: remoteResult.hasMore,
      };
    }

    const matches = normalizedQuery
      ? sourceOptions.filter((option) => option.toLowerCase().includes(normalizedQuery))
      : sourceOptions;
    const visible = matches.slice(offset, offset + MAX_RENDERED_OPTIONS);

    return {
      visible,
      total: matches.length,
      start: offset,
      hasPrevious: offset > 0,
      hasMore: offset + visible.length < matches.length,
    };
  }, [normalizedQuery, offset, open, optionKind, remoteResult, sourceOptions, totalOptions]);

  const changeQuery = (nextQuery: string) => {
    setQuery(nextQuery);
    setOffset(0);
  };

  return (
    <details className="multi-selector" open={open} onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary>{title}<span className="mono">{values.length}</span></summary>
      {open && (
        <>
          <div className="multi-search">
            <label htmlFor={searchId}>搜索条目</label>
            <div className="editorial-input">
              <Search size={14} strokeWidth={1.5} aria-hidden="true" />
              <input
                id={searchId}
                type="search"
                value={query}
                onChange={(event) => changeQuery(event.target.value)}
                placeholder={`输入关键词搜索${label}`}
                autoComplete="off"
              />
              {query && <button type="button" onClick={() => changeQuery("")} aria-label={`清除${label}搜索`}>清除</button>}
            </div>
            <span className="multi-result-count mono" aria-live="polite">
              {loadingOptions || query !== deferredQuery
                ? "正在搜索…"
                : normalizedQuery
                  ? `找到 ${remoteResult ? remoteResult.total : `${optionResult.visible.length}${optionResult.hasMore ? "+" : ""}`} 项 · 总库 ${totalOptions}`
                  : `显示 ${optionResult.start + 1}–${Math.min(optionResult.start + optionResult.visible.length, optionResult.total)} / ${optionResult.total}`}
            </span>
          </div>
          {values.length > 0 && (
            <div className="multi-selected" aria-label={`已选择 ${values.length} 项`}>
              <span className="mono">已选 {values.length}</span>
              <div>
                {values.slice(0, MAX_SELECTED_CHIPS).map((value) => (
                  <button key={value} type="button" onClick={() => onChange(values.filter((item) => item !== value))} aria-label={`移除 ${value}`}>
                    {value}<span aria-hidden="true">×</span>
                  </button>
                ))}
                {values.length > MAX_SELECTED_CHIPS && <span className="multi-selected-more mono">另有 {values.length - MAX_SELECTED_CHIPS} 项</span>}
                {values.length > 1 && <button className="multi-selected-clear" type="button" onClick={() => onChange([])}>清空全部</button>}
              </div>
            </div>
          )}
          <div className="multi-options">
            {optionResult.visible.map((option) => (
              <label className="editorial-check" key={option}>
                <input
                  type="checkbox"
                  checked={selectedSet.has(option)}
                  onChange={(event) => onChange(event.target.checked
                    ? [...values, option]
                    : values.filter((value) => value !== option))}
                />
                <span aria-hidden="true" />
                {option}
              </label>
            ))}
            {optionResult.visible.length === 0 && <p className="multi-empty">没有匹配条目，请换一个关键词。</p>}
          </div>
          {(optionResult.hasPrevious || optionResult.hasMore) && (
            <nav className="multi-pager" aria-label={`${label}分批浏览`}>
              <button type="button" disabled={!optionResult.hasPrevious || loadingOptions} onClick={() => setOffset((current) => Math.max(0, current - MAX_RENDERED_OPTIONS))}>上一批</button>
              <span className="mono">{optionResult.start + 1}–{Math.min(optionResult.start + optionResult.visible.length, optionResult.total)} / {optionResult.total}</span>
              <button type="button" disabled={!optionResult.hasMore || loadingOptions} onClick={() => setOffset((current) => current + MAX_RENDERED_OPTIONS)}>下一批</button>
            </nav>
          )}
          {children}
        </>
      )}
    </details>
  );
}

function DynamicWeights({ values, onChange, step = 0.1, fallback = 1 }: {
  values: Record<string, number>;
  onChange: (values: Record<string, number>) => void;
  step?: number;
  fallback?: number;
}) {
  const keys = Object.keys(values);
  if (!keys.length) return <p className="empty-note">尚未选择任何项目。</p>;
  return (
    <div className="dynamic-weights">
      {keys.map((key) => (
        <label key={key}>
          <span>{key}</span>
          <BufferedNumber
            label={key}
            min={0}
            max={20}
            step={step}
            value={values[key] ?? fallback}
            onCommit={(value) => onChange({ ...values, [key]: value })}
          />
        </label>
      ))}
    </div>
  );
}

export const MarginPanel = memo(function MarginPanel({ open, onToggle }: MarginPanelProps) {
  const { filters, setFilters, meta, backendStatus, catalogLoading, catalogWarnings, flash } = useAppState();
  const [coverPreview, setCoverPreview] = useState("");
  const [coverSearching, setCoverSearching] = useState(false);
  const [manualCoverError, setManualCoverError] = useState("");
  const [diagnosticOpen, setDiagnosticOpen] = useState(false);
  const [diagnosticKinds, setDiagnosticKinds] = useState<SearchCapabilityKind[]>([]);
  const [diagnosticErrors, setDiagnosticErrors] = useState<string[]>([]);
  const [diagnosticStatus, setDiagnosticStatus] = useState<SystemStatus | null>(null);
  const [diagnosticLoading, setDiagnosticLoading] = useState(false);
  const [diagnosticStatusError, setDiagnosticStatusError] = useState("");
  const semanticInputRef = useRef<HTMLInputElement>(null);
  const coverInputRef = useRef<HTMLInputElement>(null);
  const coverFileInputRef = useRef<HTMLInputElement>(null);
  const coverFileControlRef = useRef<HTMLLabelElement>(null);
  const coverSearchRequestRef = useRef(0);
  const diagnosticTriggerRef = useRef<HTMLElement | null>(null);
  const diagnosticRequestRef = useRef(0);
  const automaticWarningRef = useRef("");
  const tagOptions = meta.tags.length ? meta.tags : [...new Set(Object.keys(filters.tagWeights))].sort((a, b) => a.localeCompare(b, "zh-CN"));
  const artistOptions = meta.artists;
  const availableTitleWords = meta.titleWords.length ? meta.titleWords : titleWordOptions;
  const patchFilters = (patch: Partial<typeof filters>) => setFilters((current) => ({ ...current, ...patch }));
  const semanticWarnings = useMemo(
    () => catalogWarnings.filter((warning) => warning.startsWith("语义检索不可用：")),
    [catalogWarnings],
  );
  const coverWarnings = useMemo(
    () => catalogWarnings.filter((warning) => warning.startsWith("封面检索不可用：")),
    [catalogWarnings],
  );
  const openSearchDiagnostics = useCallback((
    kinds: SearchCapabilityKind[],
    errors: string[],
    trigger?: HTMLElement | null,
  ) => {
    diagnosticTriggerRef.current = trigger
      ?? (document.activeElement instanceof HTMLElement ? document.activeElement : null);
    setDiagnosticKinds([...new Set(kinds)]);
    setDiagnosticErrors([...new Set(errors)]);
    setDiagnosticStatus(null);
    setDiagnosticStatusError("");
    setDiagnosticLoading(true);
    setDiagnosticOpen(true);
    const requestId = diagnosticRequestRef.current + 1;
    diagnosticRequestRef.current = requestId;
    void getSystemStatus()
      .then((payload) => {
        if (diagnosticRequestRef.current === requestId) setDiagnosticStatus(payload);
      })
      .catch((error: Error) => {
        if (diagnosticRequestRef.current === requestId) setDiagnosticStatusError(error.message);
      })
      .finally(() => {
        if (diagnosticRequestRef.current === requestId) setDiagnosticLoading(false);
      });
  }, []);

  const closeSearchDiagnostics = useCallback(() => {
    setDiagnosticOpen(false);
    window.requestAnimationFrame(() => diagnosticTriggerRef.current?.focus());
  }, []);

  const automaticWarningSignature = [...semanticWarnings, ...coverWarnings].join("\n");
  useEffect(() => {
    if (!automaticWarningSignature) {
      automaticWarningRef.current = "";
      return;
    }
    if (automaticWarningRef.current === automaticWarningSignature) return;
    automaticWarningRef.current = automaticWarningSignature;
    const kinds: SearchCapabilityKind[] = [];
    if (semanticWarnings.length) kinds.push("semantic");
    if (coverWarnings.length) kinds.push("cover");
    openSearchDiagnostics(
      kinds,
      [...semanticWarnings, ...coverWarnings],
      semanticWarnings.length ? semanticInputRef.current : coverInputRef.current,
    );
  }, [automaticWarningSignature, coverWarnings, openSearchDiagnostics, semanticWarnings]);

  const setSelectedWeights = (
    key: "tagWeights" | "artistWeights" | "titleWeights",
    values: string[],
    defaultValue: number,
  ) => {
    const current = filters[key];
    patchFilters({ [key]: Object.fromEntries(values.map((value) => [value, current[value] ?? defaultValue])) });
  };

  return (
    <aside className={`margin-panel${open ? " margin-panel-open" : ""}`} aria-label="筛选与偏好设置">
      <button
        className="margin-handle"
        type="button"
        onClick={onToggle}
        aria-label={open ? "收起筛选与偏好设置" : "展开筛选与偏好设置"}
        aria-expanded={open}
      >
        {open ? <ChevronLeft size={16} /> : <ChevronRight size={16} />}
        {!open && <span>FILTER</span>}
      </button>

      <div className="margin-content" aria-hidden={!open}>
        <header className="margin-heading">
          <span className="section-code">MARGIN NOTES / 01</span>
          <h2>筛选与偏好</h2>
          <p>集中设置关键词检索、相似度查询、评分权重与最低分阈值。</p>
        </header>

        <section className="note-group">
          <span className="note-index">01 / SEARCH</span>
          <label htmlFor="editorial-keyword">实时关键词搜索</label>
          <div className="editorial-input">
            <Search size={15} strokeWidth={1.5} aria-hidden="true" />
            <input id="editorial-keyword" value={filters.keyword} onChange={(event) => patchFilters({ keyword: event.target.value })} placeholder="标题 / 译文 / 标签 / 作者" />
          </div>
          <label className="editorial-check">
            <input type="checkbox" checked={filters.keywordRelevance} onChange={(event) => patchFilters({ keywordRelevance: event.target.checked })} />
            <span aria-hidden="true" />
            启用关键词相关度
          </label>
          <p className="control-help">开启后由 MySQL FULLTEXT 计算相关度；无索引时自动回退 LIKE。</p>
        </section>

        <section className="note-group">
          <span className="note-index">02 / SIMILARITY</span>
          <label htmlFor="editorial-semantic">AI 语义检索</label>
          <div className="editorial-input">
            <BrainCircuit size={15} strokeWidth={1.5} aria-hidden="true" />
            <input
              ref={semanticInputRef}
              id="editorial-semantic"
              value={filters.semanticQuery}
              aria-invalid={semanticWarnings.length > 0}
              aria-describedby={semanticWarnings.length ? "semantic-capability-error" : undefined}
              onChange={(event) => patchFilters({ semanticQuery: event.target.value })}
              placeholder="例如：猫娘与森林"
            />
          </div>
          {semanticWarnings.length > 0 && (
            <div id="semantic-capability-error" className="capability-inline-alert" role="alert">
              <AlertTriangle size={14} aria-hidden="true" />
              <span>语义检索不可用</span>
              <button type="button" onClick={(event) => openSearchDiagnostics(["semantic"], semanticWarnings, event.currentTarget)}>查看缺失项与修复步骤</button>
            </div>
          )}
          <details className="nested-note">
            <summary>封面相似检索 · CLIP</summary>
            <label htmlFor="editorial-cover">输入库内条目 ID</label>
            <div className="editorial-input">
              <Image size={15} strokeWidth={1.5} aria-hidden="true" />
              <input
                ref={coverInputRef}
                id="editorial-cover"
                className="mono"
                value={filters.coverQuery}
                aria-invalid={coverWarnings.length > 0}
                aria-describedby={coverWarnings.length ? "cover-id-capability-error" : undefined}
                onChange={(event) => patchFilters({ coverQuery: event.target.value.toUpperCase() })}
                placeholder="JM114514 / NH123456"
              />
            </div>
            <label ref={coverFileControlRef} className="file-control" tabIndex={-1}>
              <FileImage size={15} />
              <span>{filters.coverFileName || "或上传一张图片"}</span>
              <input
                ref={coverFileInputRef}
                type="file"
                accept=".jpg,.jpeg,.png,.webp,.bmp"
                aria-invalid={Boolean(manualCoverError)}
                aria-describedby={manualCoverError ? "cover-upload-capability-error" : undefined}
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  const requestId = coverSearchRequestRef.current + 1;
                  coverSearchRequestRef.current = requestId;
                  setManualCoverError("");
                  patchFilters({ coverFileName: file?.name ?? "", coverMatches: {} });
                  if (!file) {
                    setCoverSearching(false);
                    setCoverPreview("");
                    return;
                  }
                  const reader = new FileReader();
                  reader.onload = () => {
                    if (coverSearchRequestRef.current === requestId) setCoverPreview(String(reader.result ?? ""));
                  };
                  reader.readAsDataURL(file);
                  if (backendStatus === "online") {
                    setCoverSearching(true);
                    void searchCoverFile(file).then((payload) => {
                      if (coverSearchRequestRef.current !== requestId) return;
                      patchFilters({ coverFileName: file.name, coverMatches: Object.fromEntries(payload.results.map((item) => [item.item_id, item.score])) });
                      setManualCoverError("");
                      flash(`封面向量命中 ${payload.results.length} 条候选`);
                    }).catch((error: Error) => {
                      if (coverSearchRequestRef.current !== requestId) return;
                      const message = `封面检索不可用：${error.message}`;
                      patchFilters({ coverFileName: "", coverMatches: {} });
                      setCoverPreview("");
                      setManualCoverError(message);
                      if (coverFileInputRef.current) coverFileInputRef.current.value = "";
                      openSearchDiagnostics(["cover"], [message], coverFileControlRef.current);
                    }).finally(() => {
                      if (coverSearchRequestRef.current === requestId) setCoverSearching(false);
                    });
                  }
                }}
              />
            </label>
            {coverWarnings.length > 0 && (
              <div id="cover-id-capability-error" className="capability-inline-alert" role="alert">
                <AlertTriangle size={14} aria-hidden="true" />
                <span>库内封面 ID 检索不可用</span>
                <button type="button" onClick={(event) => openSearchDiagnostics(["cover"], coverWarnings, event.currentTarget)}>查看缺失项与修复步骤</button>
              </div>
            )}
            {manualCoverError && (
              <div id="cover-upload-capability-error" className="capability-inline-alert" role="alert">
                <AlertTriangle size={14} aria-hidden="true" />
                <span>上传图片封面检索不可用</span>
                <button type="button" onClick={(event) => openSearchDiagnostics(["cover"], [manualCoverError], event.currentTarget)}>查看缺失项与修复步骤</button>
              </div>
            )}
            {coverPreview && <img className="cover-query-preview" src={coverPreview} alt="当前上传的封面查询图片" />}
            {coverSearching && <p className="control-help mono">CLIP SEARCHING…</p>}
            <p className="control-help">上传图片优先于 ID；在当前候选集内返回封面相似项。</p>
          </details>
        </section>

        <section className="note-group">
          <div className="note-title-row">
            <span className="note-index">03 / GLOBAL SCORE</span>
            <button type="button" onClick={() => patchFilters({ weights: { tag: 1, artist: 1, title: 1, history: 1 } })}>RESET</button>
          </div>
          <div className="weight-notes">
            {(Object.keys(weightLabels) as (keyof WeightState)[]).map((key) => (
              <BufferedRange
                key={key}
                label={weightLabels[key]}
                value={filters.weights[key]}
                min={0}
                max={5}
                step={0.1}
                suffix="×"
                onCommit={(value) => patchFilters({ weights: { ...filters.weights, [key]: value } })}
              />
            ))}
          </div>
          {catalogLoading && <p className="score-recalculating mono" role="status">RECALCULATING · 正在更新评分…</p>}
        </section>

        <section className="note-group preference-expanders">
          <span className="note-index">04 / PREFERENCE MAPS</span>
          <MultiSelector title="屏蔽标签配置" label="选择要屏蔽的标签" options={tagOptions} optionKind={backendStatus === "online" ? "tags" : undefined} optionTotal={meta.metrics.tags} values={filters.blockedTags} onChange={(blockedTags) => patchFilters({ blockedTags })} />
          <MultiSelector title="标签权重配置" label="加权 / 降权标签列表" options={tagOptions} optionKind={backendStatus === "online" ? "tags" : undefined} optionTotal={meta.metrics.tags} values={Object.keys(filters.tagWeights)} onChange={(values) => setSelectedWeights("tagWeights", values, 1)} defaultOpen>
            <DynamicWeights values={filters.tagWeights} onChange={(tagWeights) => patchFilters({ tagWeights })} />
          </MultiSelector>
          <MultiSelector title="作者喜爱倍数配置" label="需要特殊优待的作者" options={artistOptions} optionKind={backendStatus === "online" ? "artists" : undefined} optionTotal={meta.metrics.artists} values={Object.keys(filters.artistWeights)} onChange={(values) => setSelectedWeights("artistWeights", values, 5)}>
            <DynamicWeights values={filters.artistWeights} onChange={(artistWeights) => patchFilters({ artistWeights })} step={0.5} fallback={5} />
          </MultiSelector>
          <MultiSelector title="标题关键词权重配置" label="关键词列表" options={availableTitleWords} optionKind={backendStatus === "online" ? "titleWords" : undefined} optionTotal={meta.metrics.titleWords} values={Object.keys(filters.titleWeights)} onChange={(values) => setSelectedWeights("titleWeights", values, 1)}>
            <DynamicWeights values={filters.titleWeights} onChange={(titleWeights) => patchFilters({ titleWeights })} />
          </MultiSelector>
        </section>

        <section className="note-group note-group-last">
          <span className="note-index">05 / THRESHOLD</span>
          <BufferedRange label="最低推荐评分阈值" value={filters.minScore} min={0} max={1600} step={10} onCommit={(minScore) => patchFilters({ minScore })} />
          <p className="footnote"><sup>1</sup> {backendStatus === "online" ? "拖动时先即时预览参数，松开后只提交最终值并重算。" : "后端离线，使用内置样本即时重算。"}</p>
        </section>
      </div>
      <SearchUnavailableDialog
        open={diagnosticOpen}
        kinds={diagnosticKinds}
        errors={diagnosticErrors}
        status={diagnosticStatus}
        loading={diagnosticLoading}
        statusError={diagnosticStatusError}
        onClose={closeSearchDiagnostics}
      />
    </aside>
  );
});

MarginPanel.displayName = "MarginPanel";
