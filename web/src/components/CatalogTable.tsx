import {
  Fragment,
  memo,
  startTransition,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { ArrowUpRight, Check, FolderOpen } from "lucide-react";
import { Link } from "react-router-dom";
import { getGallery } from "../api/client";
import type { CatalogItem } from "../data/catalog";

export type DisplayItem = CatalogItem & { score: number };

type CatalogTableProps = {
  rows: DisplayItem[];
  rowOffset?: number;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  onOpenSource?: (id: string) => void;
  onPreview?: (item: DisplayItem | null) => void;
  showKeywordRelevance: boolean;
  showAiRelevance: boolean;
  showCoverRelevance: boolean;
};

const ROW_HEIGHT = 52;
const ROW_OVERSCAN = 8;
const DETAIL_HEIGHT_ESTIMATE = 440;

export function EditorialCover({
  item,
  large = false,
  transitionName,
}: {
  item: CatalogItem;
  large?: boolean;
  transitionName?: string;
}) {
  const [imageFailed, setImageFailed] = useState(false);
  if (item.coverUrl && !imageFailed) {
    return (
      <img
        className={`editorial-cover${large ? " editorial-cover-large" : ""}`}
        src={item.coverUrl}
        alt={`${item.titleZh || item.title}封面`}
        loading="lazy"
        onError={() => setImageFailed(true)}
        style={transitionName ? { viewTransitionName: transitionName } as React.CSSProperties : undefined}
      />
    );
  }
  return (
    <svg
      className={`editorial-cover${large ? " editorial-cover-large" : ""}`}
      viewBox="0 0 96 128"
      role="img"
      aria-label={`${item.titleZh}封面占位图`}
      style={transitionName ? { viewTransitionName: transitionName } as React.CSSProperties : undefined}
    >
      <rect width="96" height="128" fill="var(--panel)" />
      <rect x="7" y="7" width="82" height="114" fill="var(--paper)" stroke="var(--ink)" />
      {item.coverCode === "circle" && (
        <>
          <circle cx="48" cy="50" r="26" fill="var(--ink)" />
          <circle cx="48" cy="50" r="12" fill="var(--paper)" />
        </>
      )}
      {item.coverCode === "slash" && (
        <>
          <path d="M18 98 L70 22 H83 L31 98 Z" fill="var(--ink)" />
          <path d="M14 105 H82" stroke="var(--ink-3)" />
        </>
      )}
      {item.coverCode === "frame" && (
        <>
          <rect x="20" y="24" width="56" height="64" fill="var(--ink)" />
          <rect x="28" y="32" width="40" height="48" fill="var(--paper)" />
          <rect x="36" y="40" width="24" height="32" fill="var(--ink-3)" />
        </>
      )}
      {item.coverCode === "type" && (
        <>
          <text x="16" y="52" fill="var(--ink)" fontSize="26" fontWeight="800">XP</text>
          <path d="M16 62 H80 M16 70 H62 M16 78 H72" stroke="var(--ink)" strokeWidth="2" />
        </>
      )}
      <text x="13" y="112" fill="var(--ink)" fontSize="7" fontFamily="monospace">
        {item.id}
      </text>
    </svg>
  );
}

function Score({ value }: { value: number }) {
  return <span className="signal-score mono">{value}</span>;
}

export const CatalogCoverPreview = memo(function CatalogCoverPreview({ item }: { item: DisplayItem | null }) {
  return (
    <aside className="cover-preview" aria-label="库存封面大图预览">
      <span className="preview-kicker mono">COVER PREVIEW / HOVER OR FOCUS</span>
      {item ? (
        <div className="cover-reveal" key={item.id}>
          <EditorialCover item={item} large />
          <div className="preview-caption">
            <span className="preview-number mono">{item.id}</span>
            <h3>{item.titleZh || item.title}</h3>
            <p>{item.titleZh && item.title ? item.title : item.artist || "—"}</p>
            <dl>
              <div><dt>作者</dt><dd>{item.artist || "—"}</dd></div>
              <div><dt>团队</dt><dd>{item.circle || "—"}</dd></div>
              <div><dt>语言</dt><dd>{item.language || "—"}</dd></div>
              <div><dt>页数</dt><dd className="mono">{item.pages}</dd></div>
              <div><dt>推荐分</dt><dd className="mono signal-text">{item.score}</dd></div>
            </dl>
          </div>
        </div>
      ) : (
        <div className="cover-preview-empty">
          <span className="mono">NO PREVIEW</span>
          <p>将鼠标移到目录行，或用键盘聚焦行内操作，即可查看封面大图。</p>
        </div>
      )}
    </aside>
  );
});

CatalogCoverPreview.displayName = "CatalogCoverPreview";

function ExpandedCatalogDetails({
  item,
  onOpenSource,
  onHeightChange,
}: {
  item: DisplayItem;
  onOpenSource?: (id: string) => void;
  onHeightChange?: (height: number) => void;
}) {
  const [remoteItem, setRemoteItem] = useState<(DisplayItem & { titleWords?: string[]; rawTags?: string[] }) | null>(null);
  const [loading, setLoading] = useState(true);
  const sectionRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const controller = new AbortController();
    setRemoteItem(null);
    setLoading(true);
    void getGallery(item.id, controller.signal)
      .then((payload) => setRemoteItem({ ...payload, score: item.score }))
      .catch(() => setRemoteItem(null))
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [item.id]);

  useEffect(() => {
    const node = sectionRef.current;
    if (!node || !onHeightChange) return;
    const report = () => onHeightChange(node.offsetHeight);
    report();
    const observer = new ResizeObserver(report);
    observer.observe(node);
    return () => observer.disconnect();
  }, [onHeightChange]);

  const current = remoteItem ? { ...item, ...remoteItem, score: item.score } : item;
  const details = [
    ["条目 ID", current.id],
    ["作者", current.artist || "—"],
    ["团队 / 社团", current.circle || "—"],
    ["语言", current.language || "—"],
    ["页数", `${current.pages} 页`],
    ["上传日期", current.uploadedAt || "—"],
    ["推荐评分", String(current.score)],
    ["基础评分", String(current.baseScore)],
    ["关键词相关度", current.keywordRelevance.toFixed(2)],
    ["AI 相关度", current.aiRelevance.toFixed(2)],
    ["封面相关度", current.coverRelevance.toFixed(2)],
    ["标题特征词", remoteItem?.titleWords?.join("、") || "—"],
    ["文件名", current.filename || "—"],
    ["本地目录", current.localPath || "—"],
    ["网络来源", current.link || "未记录"],
  ];

  return (
    <section ref={sectionRef} id={`catalog-detail-${item.id}`} className="expanded-record" aria-label={`${current.titleZh || current.title}完整信息`}>
      <div className="expanded-record-lead">
        <EditorialCover item={current} large transitionName={`cover-${item.id}`} />
        <div className="expanded-record-copy">
          <span className="teaser-label mono">FULL RECORD / {item.id}</span>
          <h3>{current.titleZh || current.title}</h3>
          {current.titleZh && current.title && <p className="expanded-original">{current.title}</p>}
          <p className="expanded-summary">{current.summary || "暂无内容摘要。"}</p>
          {loading && <span className="expanded-loading mono" role="status">LOADING FULL RECORD…</span>}
          <div className="expanded-actions">
            <Link to={`/detail/${item.id}`} viewTransition>进入漫画详情<ArrowUpRight size={14} /></Link>
            <a
              href={current.link ? `/api/track/${encodeURIComponent(item.id)}` : `https://example.com/gallery/${item.id}`}
              target="_blank"
              rel="noreferrer"
              onClick={() => { if (!item.link) onOpenSource?.(item.id); }}
            >打开网络来源<ArrowUpRight size={14} /></a>
          </div>
        </div>
      </div>

      <dl className="expanded-record-grid">
        {details.map(([label, value]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd className={label === "本地目录" || label === "网络来源" || label === "文件名" || label === "条目 ID" ? "mono" : undefined}>{value}</dd>
          </div>
        ))}
      </dl>

      <div className="expanded-tags">
        <span>完整标签</span>
        <p>{current.tags.length ? current.tags.join("、") : "—"}</p>
      </div>
      <span className="teaser-path mono"><FolderOpen size={13} />{current.localPath || "未记录本地目录"}</span>
    </section>
  );
}

export const CatalogTable = memo(function CatalogTable({
  rows,
  rowOffset = 0,
  selectedId,
  onSelect,
  onOpenSource,
  onPreview,
  showKeywordRelevance,
  showAiRelevance,
  showCoverRelevance,
}: CatalogTableProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const scrollFrameRef = useRef<number | null>(null);
  const pendingScrollTopRef = useRef(0);
  const [scrollTop, setScrollTop] = useState(0);
  const [viewportHeight, setViewportHeight] = useState(650);
  const [detailHeight, setDetailHeight] = useState(DETAIL_HEIGHT_ESTIMATE);
  const relevanceCount = Number(showCoverRelevance)
    + Number(showAiRelevance)
    + Number(showKeywordRelevance);
  const columnCount = 14 + relevanceCount;
  const relevanceWidth = 4.5;
  const baseScale = (100 - relevanceCount * relevanceWidth) / 100;
  const baseColumnWidths = [5, 4, 6, 12, 12, 7, 7, 13, 5.5, 4, 6.5, 9, 4];
  const selectedIndex = useMemo(() => rows.findIndex((item) => item.id === selectedId), [rows, selectedId]);
  const firstRowId = rows[0]?.id ?? "";

  useEffect(() => {
    const node = containerRef.current;
    if (!node) return;
    node.scrollTop = 0;
    pendingScrollTopRef.current = 0;
    setScrollTop(0);
  }, [firstRowId, rowOffset]);

  useEffect(() => {
    const node = containerRef.current;
    if (!node) return;
    const updateHeight = () => setViewportHeight(node.clientHeight || 650);
    updateHeight();
    const observer = new ResizeObserver(updateHeight);
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    setDetailHeight(selectedId ? DETAIL_HEIGHT_ESTIMATE : 0);
  }, [selectedId]);

  useEffect(() => () => {
    if (scrollFrameRef.current !== null) window.cancelAnimationFrame(scrollFrameRef.current);
  }, []);

  const handleScroll = useCallback((event: React.UIEvent<HTMLDivElement>) => {
    onPreview?.(null);
    pendingScrollTopRef.current = event.currentTarget.scrollTop;
    if (scrollFrameRef.current !== null) return;
    scrollFrameRef.current = window.requestAnimationFrame(() => {
      scrollFrameRef.current = null;
      startTransition(() => setScrollTop(pendingScrollTopRef.current));
    });
  }, [onPreview]);

  const handleDetailHeight = useCallback((height: number) => {
    setDetailHeight((current) => Math.abs(current - height) > 1 ? height : current);
  }, []);

  const activeDetailHeight = selectedIndex >= 0 ? detailHeight : 0;
  const detailStart = selectedIndex * ROW_HEIGHT + ROW_HEIGHT;
  let dataScrollTop = scrollTop;
  if (selectedIndex >= 0 && scrollTop >= detailStart) {
    dataScrollTop = scrollTop < detailStart + activeDetailHeight
      ? selectedIndex * ROW_HEIGHT
      : scrollTop - activeDetailHeight;
  }
  const visibleRowCount = Math.ceil(viewportHeight / ROW_HEIGHT) + ROW_OVERSCAN * 2 + 2;
  const startIndex = Math.max(0, Math.floor(dataScrollTop / ROW_HEIGHT) - ROW_OVERSCAN);
  const endIndex = Math.min(rows.length, startIndex + visibleRowCount);
  const visibleRows = rows.slice(startIndex, endIndex);
  const topSpacerHeight = startIndex * ROW_HEIGHT
    + (selectedIndex >= 0 && selectedIndex < startIndex ? activeDetailHeight : 0);
  const bottomSpacerHeight = (rows.length - endIndex) * ROW_HEIGHT
    + (selectedIndex >= endIndex ? activeDetailHeight : 0);

  return (
    <div ref={containerRef} className="catalog-table-scroll" tabIndex={0} aria-label="库存目录表格" onScroll={handleScroll}>
      <table className="catalog-table" aria-rowcount={rows.length + 1}>
        <colgroup>
          {baseColumnWidths.map((width, index) => <col key={`base-${index}`} style={{ width: `${width * baseScale}%` }} />)}
          {Array.from({ length: relevanceCount }, (_, index) => <col key={`relevance-${index}`} style={{ width: `${relevanceWidth}%` }} />)}
          <col style={{ width: `${5 * baseScale}%` }} />
        </colgroup>
        <thead>
          <tr>
            <th className="index-col">№</th>
            <th className="cover-col">封面</th>
            <th className="id-col">ID</th>
            <th className="translation-col">标题译文</th>
            <th className="original-col">原标题</th>
            <th className="artist-col">作者</th>
            <th className="circle-col">团队</th>
            <th className="tags-col">标签</th>
            <th className="language-col">语言</th>
            <th className="pages-col number-col">页数</th>
            <th className="date-col">上传日期</th>
            <th className="path-col">本地目录</th>
            <th className="link-col">链接</th>
            {showCoverRelevance && <th className="relevance-col number-col">封面相关</th>}
            {showAiRelevance && <th className="relevance-col number-col">AI 相关</th>}
            {showKeywordRelevance && <th className="relevance-col number-col">关键词相关</th>}
            <th className="score-col number-col">推荐评分</th>
          </tr>
        </thead>
        <tbody>
          {topSpacerHeight > 0 && (
            <tr className="catalog-spacer-row" aria-hidden="true">
              <td colSpan={columnCount} style={{ height: topSpacerHeight }} />
            </tr>
          )}
          {visibleRows.map((item, visibleIndex) => {
            const index = startIndex + visibleIndex;
            const selected = selectedId === item.id;
            return (
              <Fragment key={item.id}>
                <tr
                  className={`catalog-row${selected ? " catalog-row-selected" : ""}`}
                  aria-rowindex={index + 2}
                  onPointerEnter={() => onPreview?.(item)}
                  onPointerLeave={(event) => {
                    if (!event.currentTarget.contains(document.activeElement)) onPreview?.(null);
                  }}
                  onFocusCapture={() => onPreview?.(item)}
                  onBlurCapture={(event) => {
                    const nextTarget = event.relatedTarget as Node | null;
                    if (!nextTarget || !event.currentTarget.contains(nextTarget)) onPreview?.(null);
                  }}
                >
                  <td className="index-col">
                    <button
                      className="index-button mono"
                      type="button"
                      onClick={() => onSelect(selected ? null : item.id)}
                      aria-label={selected ? `取消选择${item.titleZh}` : `选择${item.titleZh}`}
                      aria-pressed={selected}
                      aria-expanded={selected}
                      aria-controls={`catalog-detail-${item.id}`}
                    >
                      <i aria-hidden="true" />
                      {selected ? <Check size={13} strokeWidth={1.7} /> : `№ ${String(rowOffset + index + 1).padStart(4, "0")}`}
                    </button>
                  </td>
                  <td className="cover-col"><EditorialCover item={item} transitionName={selected ? `cover-${item.id}` : undefined} /></td>
                  <td className="id-col mono">{item.id}</td>
                  <td className="translation-col row-title" style={{ viewTransitionName: `title-${item.id}` } as React.CSSProperties}>{item.titleZh}</td>
                  <td className="original-col row-original">{item.title}</td>
                  <td className="artist-col">{item.artist}</td>
                  <td className="circle-col">{item.circle}</td>
                  <td className="tags-col">{item.tags.join("，")}</td>
                  <td className="language-col">{item.language}</td>
                  <td className="pages-col number-col mono">{item.pages}</td>
                  <td className="date-col mono">{item.uploadedAt}</td>
                  <td className="path-col mono">{item.localPath}</td>
                  <td className="link-col">
                    <a
                      href={item.link ? `/api/track/${encodeURIComponent(item.id)}` : `https://example.com/gallery/${item.id}`}
                      target="_blank"
                      rel="noreferrer"
                      onClick={() => { if (!item.link) onOpenSource?.(item.id); }}
                    >打开</a>
                  </td>
                  {showCoverRelevance && <td className="relevance-col number-col mono">{item.coverRelevance.toFixed(2)}</td>}
                  {showAiRelevance && <td className="relevance-col number-col mono">{item.aiRelevance.toFixed(2)}</td>}
                  {showKeywordRelevance && <td className="relevance-col number-col mono">{item.keywordRelevance.toFixed(2)}</td>}
                  <td className="score-col number-col"><Score value={item.score} /></td>
                </tr>
                {selected && (
                  <tr className="teaser-row expanded-record-row">
                    <td colSpan={columnCount}>
                      <ExpandedCatalogDetails item={item} onOpenSource={onOpenSource} onHeightChange={handleDetailHeight} />
                    </td>
                  </tr>
                )}
              </Fragment>
            );
          })}
          {bottomSpacerHeight > 0 && (
            <tr className="catalog-spacer-row" aria-hidden="true">
              <td colSpan={columnCount} style={{ height: bottomSpacerHeight }} />
            </tr>
          )}
        </tbody>
      </table>
      {rows.length === 0 && (
        <div className="catalog-empty">
          <span className="mono">INDEX / 0000</span>
          <h3>本页没有匹配条目</h3>
          <p>请调整页边注中的检索词或评分阈值。</p>
        </div>
      )}
    </div>
  );
});

CatalogTable.displayName = "CatalogTable";
