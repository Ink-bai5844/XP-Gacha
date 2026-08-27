import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowDown,
  ChevronDown,
  Clipboard,
  Eye,
  EyeOff,
  RefreshCw,
} from "lucide-react";
import { CatalogCoverPreview, CatalogTable, type DisplayItem } from "../components/CatalogTable";
import { CatalogImportGuide } from "../components/CatalogImportGuide";
import { MarginPanel } from "../components/MarginPanel";
import { refreshCovers } from "../api/client";
import { useAppState, useCatalogResults, type SortKey } from "../state/AppState";

const COVER_PREVIEW_STORAGE_KEY = "xp-gacha.catalog.cover-preview";

function initialCoverPreviewState() {
  try {
    const stored = window.localStorage.getItem(COVER_PREVIEW_STORAGE_KEY);
    if (stored === "on" || stored === "off") return stored === "on";
  } catch {
    // Fall through to the viewport-aware default.
  }
  return typeof window.matchMedia === "function"
    ? window.matchMedia("(min-width: 1440px)").matches
    : window.innerWidth >= 1440;
}

const sortLabels: Record<SortKey, string> = {
  score: "推荐评分",
  keyword: "关键词相关度",
  semantic: "AI 相关度",
  cover: "封面相关度",
  id: "ID",
  date: "上传日期",
  titleZh: "标题译文",
  title: "标题",
  artist: "作者",
  circle: "团队",
  tags: "标签",
  language: "语言",
  pages: "页数",
  localPath: "本地目录",
};

type CatalogWorkspaceProps = {
  rows: DisplayItem[];
  rowOffset: number;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  onOpenSource: (id: string) => void;
  previewOpen: boolean;
  showKeywordRelevance: boolean;
  showAiRelevance: boolean;
  showCoverRelevance: boolean;
};

const CatalogWorkspace = memo(function CatalogWorkspace({
  rows,
  rowOffset,
  selectedId,
  onSelect,
  onOpenSource,
  previewOpen,
  showKeywordRelevance,
  showAiRelevance,
  showCoverRelevance,
}: CatalogWorkspaceProps) {
  const [hoveredItem, setHoveredItem] = useState<DisplayItem | null>(null);
  const previewFrameRef = useRef<number | null>(null);
  const selectedItem = useMemo(
    () => rows.find((item) => item.id === selectedId) ?? null,
    [rows, selectedId],
  );

  const handlePreview = useCallback((item: DisplayItem | null) => {
    if (previewFrameRef.current !== null) window.cancelAnimationFrame(previewFrameRef.current);
    previewFrameRef.current = window.requestAnimationFrame(() => {
      previewFrameRef.current = null;
      setHoveredItem(item);
    });
  }, []);

  useEffect(() => () => {
    if (previewFrameRef.current !== null) window.cancelAnimationFrame(previewFrameRef.current);
  }, []);

  useEffect(() => {
    if (!previewOpen) setHoveredItem(null);
  }, [previewOpen]);

  useEffect(() => {
    setHoveredItem((current) => current
      ? rows.find((item) => item.id === current.id) ?? null
      : null);
  }, [rows]);

  return (
    <div className={`catalog-spread${previewOpen ? " catalog-spread-with-preview" : ""}`}>
      <CatalogTable
        rows={rows}
        rowOffset={rowOffset}
        selectedId={selectedId}
        onSelect={onSelect}
        onOpenSource={onOpenSource}
        onPreview={previewOpen ? handlePreview : undefined}
        showKeywordRelevance={showKeywordRelevance}
        showAiRelevance={showAiRelevance}
        showCoverRelevance={showCoverRelevance}
      />
      {previewOpen && <CatalogCoverPreview item={hoveredItem ?? selectedItem} />}
    </div>
  );
});

CatalogWorkspace.displayName = "CatalogWorkspace";

export function CatalogPage() {
  const {
    filters,
    sortKey,
    setSortKey,
    descending,
    setDescending,
    page,
    setPage,
    selectedId,
    setSelectedId,
    recordHistory,
    flash,
    backendStatus,
    databaseAvailable,
    catalogTotal,
    catalogMetrics,
    catalogLoading,
    catalogReady,
    catalogWarnings,
    recall,
    pageSize,
    refreshLibrary,
  } = useAppState();
  const rows = useCatalogResults();
  const [panelOpen, setPanelOpen] = useState(true);
  const [queueCount, setQueueCount] = useState(0);
  const [coverPreviewOpen, setCoverPreviewOpen] = useState(initialCoverPreviewState);
  const handleOpenSource = useCallback((id: string) => recordHistory(id, "打开网络来源"), [recordHistory]);
  const handlePanelToggle = useCallback(() => setPanelOpen((current) => !current), []);

  useEffect(() => {
    try {
      window.localStorage.setItem(COVER_PREVIEW_STORAGE_KEY, coverPreviewOpen ? "on" : "off");
    } catch {
      // Storage can be unavailable in privacy-restricted browser contexts.
    }
  }, [coverPreviewOpen]);

  const filterSignature = useMemo(() => JSON.stringify(filters), [filters]);
  useEffect(() => setPage(0), [filterSignature, setPage]);

  const totalPages = Math.max(1, Math.ceil(catalogTotal / pageSize));
  useEffect(() => {
    if (page >= totalPages) setPage(totalPages - 1);
  }, [page, setPage, totalPages]);

  const pageRows = rows;
  const databaseEmpty = backendStatus === "online" && databaseAvailable === true && catalogReady && catalogMetrics.items === 0;

  const availableSorts = useMemo<SortKey[]>(() => {
    const optional: SortKey[] = [];
    if (filters.semanticQuery) optional.push("semantic");
    if (filters.coverQuery || filters.coverFileName) optional.push("cover");
    if (filters.keywordRelevance) optional.push("keyword");
    return [...optional, "score", "id", "date", "titleZh", "title", "artist", "circle", "tags", "language", "pages", "localPath"];
  }, [filters.coverFileName, filters.coverQuery, filters.keywordRelevance, filters.semanticQuery]);

  useEffect(() => {
    if (!availableSorts.includes(sortKey)) setSortKey("score");
  }, [availableSorts, setSortKey, sortKey]);

  const copyPage = async () => {
    const text = pageRows.map((item, index) => [
      `№ ${String(page * pageSize + index + 1).padStart(4, "0")}`,
      item.id,
      item.titleZh,
      item.title,
      item.artist,
      item.circle,
      item.tags.join(", "),
      item.language,
      item.pages,
      item.uploadedAt,
      item.localPath,
      item.score,
    ].join("\t")).join("\n");
    try {
      await navigator.clipboard.writeText(text);
      flash(`当前页 ${pageRows.length} 条目录已复制`);
    } catch {
      flash("浏览器未允许写入剪贴板");
    }
  };

  const start = catalogTotal ? page * pageSize : 0;
  const end = Math.min((page + 1) * pageSize - 1, Math.max(0, catalogTotal - 1));

  return (
    <div className={`catalog-layout${panelOpen ? " catalog-layout-panel-open" : ""}`}>
      <MarginPanel open={panelOpen} onToggle={handlePanelToggle} />

      <section className="catalog-content" aria-label="库存目录工作区">
        <div className="metrics-ledger" aria-label="库存摘要">
          <article><span className="metric-code mono">CURRENT SELECTION</span><strong className="mono">{catalogTotal.toLocaleString()}</strong><p>当前筛选条目 / 册</p></article>
          <article><span className="metric-code mono">ARTISTS</span><strong className="mono">{catalogMetrics.artists.toLocaleString()}</strong><p>总收录作者 / 位</p></article>
          <article><span className="metric-code mono">TAXONOMY</span><strong className="mono">{catalogMetrics.tags.toLocaleString()}</strong><p>总标签种类 / 种</p></article>
          <article><span className="metric-code mono">TITLE TOKENS</span><strong className="mono">{catalogMetrics.titleWords.toLocaleString()}</strong><p>解析标题词汇 / 种</p></article>
        </div>

        <header className="catalog-heading">
          <div><span className="section-code">SECTION 01 / LIBRARY</span><h2>库存目录</h2></div>
          <p>整合 MySQL 关键词召回、动态评分、语义检索与封面相似检索，集中浏览和筛选库存。<sup>01</sup></p>
        </header>

        {databaseEmpty ? (
          <CatalogImportGuide />
        ) : <>
        {filters.keyword && (
          <p className="recall-caption mono">关键词召回：{recall.mode} · {recall.candidateCount} 个候选 · {filters.keywordRelevance ? "已使用全文相关度" : "仅候选召回"}</p>
        )}

        {catalogWarnings
          .filter((warning) => !warning.startsWith("语义检索不可用：") && !warning.startsWith("封面检索不可用："))
          .map((warning) => <p className="recall-caption mono" role="status" key={warning}>{warning}</p>)}

        <div className="catalog-toolbar" aria-label="库存表格工具栏">
          <span className="toolbar-folio mono">VOL.2026 — 结果 {catalogTotal} 条 — 第 {page + 1}/{totalPages} 页{catalogLoading ? " · LOADING" : ""}</span>
          <label>
            <span>SORT</span>
            <select value={sortKey} onChange={(event) => setSortKey(event.target.value as SortKey)} aria-label="全局排序依据">
              {availableSorts.map((key) => <option value={key} key={key}>{sortLabels[key]}</option>)}
            </select>
            <ChevronDown size={13} aria-hidden="true" />
          </label>
          <button type="button" onClick={() => setDescending((current) => !current)} aria-label={descending ? "当前降序，点击切换升序" : "当前升序，点击切换降序"}>
            <ArrowDown size={13} className={descending ? "" : "rotate-arrow"} />{descending ? "DESC" : "ASC"}
          </button>
          <label>
            <span>RANGE</span>
            <select value={page} onChange={(event) => setPage(Number(event.target.value))} aria-label="选择显示范围">
              {Array.from({ length: totalPages }, (_, index) => {
                const rangeStart = catalogTotal ? index * pageSize : 0;
                const rangeEnd = Math.min((index + 1) * pageSize - 1, Math.max(0, catalogTotal - 1));
                return <option value={index} key={index}>{rangeStart} ~ {rangeEnd}</option>;
              })}
            </select>
            <ChevronDown size={13} aria-hidden="true" />
          </label>
          <button type="button" onClick={copyPage}><Clipboard size={13} />复制当前页 {pageRows.length} 条</button>
          <button type="button" onClick={() => {
            const ids = pageRows.map((item) => item.id);
            if (backendStatus === "online") void refreshCovers(ids).then(({ queued }) => { setQueueCount(queued); flash(`已提交 ${queued} 个封面检查`); window.setTimeout(() => { setQueueCount(0); refreshLibrary(); }, 1200); }).catch((error: Error) => flash(error.message));
            else flash("演示模式不抓取在线封面");
          }}>
            <RefreshCw size={13} />刷新封面{queueCount > 0 && <sup className="queue-count mono">{queueCount}</sup>}
          </button>
          <button
            className="cover-preview-toggle"
            type="button"
            aria-pressed={coverPreviewOpen}
            onClick={() => setCoverPreviewOpen((current) => !current)}
          >
            {coverPreviewOpen ? <EyeOff size={13} /> : <Eye size={13} />}
            {coverPreviewOpen ? "收起大图预览" : "展开大图预览"}
          </button>
        </div>

        {(filters.keywordRelevance || filters.semanticQuery || filters.coverQuery || filters.coverFileName || filters.blockedTags.length > 0) && (
          <div className="filter-summary" role="status">
            <span className="mono">ACTIVE FILTERS</span>
            {filters.keywordRelevance && <b>关键词相关度</b>}
            {filters.semanticQuery && <b>语义：{filters.semanticQuery}</b>}
            {(filters.coverQuery || filters.coverFileName) && <b>封面：{filters.coverFileName || filters.coverQuery}</b>}
            {filters.blockedTags.length > 0 && <b>屏蔽：{filters.blockedTags.join("、")}</b>}
          </div>
        )}

        <CatalogWorkspace
          rows={pageRows}
          rowOffset={page * pageSize}
          selectedId={selectedId}
          onSelect={setSelectedId}
          onOpenSource={handleOpenSource}
          previewOpen={coverPreviewOpen}
          showKeywordRelevance={filters.keywordRelevance}
          showAiRelevance={Boolean(filters.semanticQuery)}
          showCoverRelevance={Boolean(filters.coverQuery || filters.coverFileName)}
        />
        </>}
      </section>

      {!databaseEmpty && (
        <div className="catalog-postscript">
          {queueCount > 0 && <p className="pending-caption mono">有 {queueCount} 个封面正在后台抓取；完成后点击「刷新封面」查看。</p>}
          <footer className="catalog-footnotes">
            <p><sup>01</sup> 默认排序：推荐评分降序，其次按上传日期降序。当前范围 {start} ~ {end}。</p>
            <p><sup>02</sup> 运行模式：{backendStatus === "online" ? "API / MySQL 实际数据" : "离线演示数据"}。</p>
          </footer>
        </div>
      )}
    </div>
  );
}
