import { useEffect, useMemo, useRef, useState } from "react";
import { getCharts, type ChartPayload } from "../api/client";
import { catalogItems } from "../data/catalog";
import { findCatalogItem, titleWordOptions, useAppState } from "../state/AppState";

type RankItem = { label: string; value: number };
type ChartSet = { title: string; labelName: string; valueName: string; items: RankItem[] };

function rank(values: string[]) {
  const counts = new Map<string, number>();
  values.forEach((value) => counts.set(value, (counts.get(value) ?? 0) + 1));
  return [...counts].map(([label, value]) => ({ label, value })).sort((a, b) => b.value - a.value || a.label.localeCompare(b.label, "zh-CN"));
}

function RankTable({ set }: { set: ChartSet }) {
  const [open, setOpen] = useState(false);
  return (
    <details className="rank-table" open={open} onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary>查看 Top 150 {set.labelName}<span className="mono">{set.items.length}</span></summary>
      {open && (
        <table>
          <thead><tr><th>№</th><th>{set.labelName}</th><th>{set.valueName}</th></tr></thead>
          <tbody>{set.items.slice(0, 150).map((item, index) => <tr key={item.label}><td className="mono">{String(index + 1).padStart(3, "0")}</td><td>{item.label}</td><td className="mono number-col">{item.value}</td></tr>)}</tbody>
        </table>
      )}
    </details>
  );
}

function BarJournal({ set }: { set: ChartSet }) {
  const max = Math.max(1, ...set.items.map((item) => item.value));
  const first = set.items[0];
  return (
    <article className="journal-chart">
      <span className="chart-kicker mono">TOP 15 / {set.labelName.toUpperCase()}</span>
      <h3>{first ? `${first.label}位居首位` : "尚无足够记录"}</h3>
      <p>{set.title}。数值始终直接标注，并提供展开数据表。</p>
      <div className="bar-list" role="img" aria-label={set.title}>
        {set.items.slice(0, 15).map((item, index) => (
          <div className={`bar-row${index === 0 ? " bar-row-first" : ""}`} key={item.label} aria-label={`${item.label} ${item.value}`}>
            <span>{item.label}</span>
            <i><b style={{ width: `${(item.value / max) * 100}%` }} /></i>
            <strong className="mono">{item.value}</strong>
          </div>
        ))}
        {!set.items.length && <div className="chart-empty">等待历史记录产生统计数据。</div>}
      </div>
      <RankTable set={set} />
    </article>
  );
}

export function PreferenceCharts({ scope = "global", compact = false }: { scope?: "global" | "history"; compact?: boolean }) {
  const { history, backendStatus } = useAppState();
  const [remote, setRemote] = useState<ChartPayload | null>(null);
  const [active, setActive] = useState(!compact);
  const sectionRef = useRef<HTMLElement>(null);

  useEffect(() => {
    if (!compact) {
      setActive(true);
      return;
    }
    const node = sectionRef.current;
    if (!node || typeof IntersectionObserver === "undefined") {
      setActive(true);
      return;
    }
    const observer = new IntersectionObserver(([entry]) => {
      if (!entry.isIntersecting) return;
      setActive(true);
      observer.disconnect();
    }, { rootMargin: "120px" });
    observer.observe(node);
    return () => observer.disconnect();
  }, [compact]);

  useEffect(() => {
    if (!active || backendStatus !== "online") return;
    const controller = new AbortController();
    void getCharts(scope, controller.signal).then(setRemote).catch(() => setRemote(null));
    return () => controller.abort();
  }, [active, backendStatus, history.length, scope]);

  const sets = useMemo<ChartSet[]>(() => {
    if (!active) return [];
    if (remote) {
      return ["tags", "artists", "title_words"].map((key) => {
        const set = remote[key];
        return {
          title: set?.title ?? key,
          labelName: set?.label_col ?? key,
          valueName: set?.value_col ?? "频次",
          items: set?.top_150 ?? [],
        };
      });
    }
    const items = scope === "global"
      ? catalogItems
      : history.map((entry) => findCatalogItem(entry.itemId)).filter((item): item is NonNullable<typeof item> => Boolean(item));
    const prefix = scope === "global" ? "" : "历史偏好";
    return [
      { title: `Top 15 ${prefix || "XP"} 标签分布`, labelName: `${prefix}标签`, valueName: scope === "global" ? "出现频次" : "打开频次", items: rank(items.flatMap((item) => item.tags)) },
      { title: `Top 15 ${prefix || "核心"} 作者分布`, labelName: `${prefix}作者`, valueName: scope === "global" ? "收录册数" : "打开频次", items: rank(items.map((item) => item.artist)) },
      { title: `Top 15 ${prefix || "标题"} 高频词汇`, labelName: `${prefix}标题词`, valueName: scope === "global" ? "出现频次" : "打开频次", items: rank(items.flatMap((item) => titleWordOptions.filter((word) => item.titleZh.includes(word)))) },
    ];
  }, [active, history, remote, scope]);

  return (
    <section ref={sectionRef} className={`preference-section${compact ? " preference-section-compact" : ""}`}>
      <header>
        <span className="section-code">{scope === "global" ? "DATASET / GLOBAL" : "DATASET / RECOMMENDATION HISTORY"}</span>
        <h2>{scope === "global" ? "全局偏好数据" : "用户历史偏好数据"}</h2>
        {scope === "history" && <p className="mono">数据来源：datacache/recommendation_history.json · 当前 {history.length} 条</p>}
      </header>
      {active
        ? <div className="journal-chart-grid">{sets.map((set) => <BarJournal set={set} key={set.title} />)}</div>
        : <div className="chart-lazy-placeholder mono">滚动至此处时载入图表…</div>}
    </section>
  );
}
