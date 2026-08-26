import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type Dispatch,
  type ReactNode,
  type SetStateAction,
} from "react";
import {
  addHistory,
  getHealth,
  getHistory,
  getMeta,
  queryLibrary,
  removeAllHistory,
  removeHistory,
  type BackendStatus,
  type LibraryMetrics,
  type MetaOptions,
} from "../api/client";
import { catalogItems, type CatalogItem } from "../data/catalog";

export type WeightState = { tag: number; artist: number; title: number; history: number };

export type FilterState = {
  keyword: string;
  keywordRelevance: boolean;
  semanticQuery: string;
  coverQuery: string;
  coverFileName: string;
  coverMatches: Record<string, number>;
  weights: WeightState;
  minScore: number;
  blockedTags: string[];
  tagWeights: Record<string, number>;
  artistWeights: Record<string, number>;
  titleWeights: Record<string, number>;
};

export type SortKey = "score" | "keyword" | "semantic" | "cover" | "id" | "date" | "titleZh" | "title" | "artist" | "circle" | "tags" | "language" | "pages" | "localPath";
export type HistoryAction = "打开本地目录" | "打开网络来源";
export type HistoryEntry = {
  key: string;
  openedAt: string;
  action: HistoryAction;
  itemId: string;
  title?: string;
  author?: string;
  link?: string;
  localPath?: string;
  tags?: string[];
  titleWords?: string[];
};
export type ChatMessage = { id: string; role: "user" | "assistant"; content: string; thinking?: string; citations?: string[] };
export type ScoredCatalogItem = CatalogItem & { score: number };

type RecallState = { mode: string; candidateCount: number; usedFulltext: boolean };
type AppStateValue = {
  filters: FilterState;
  setFilters: Dispatch<SetStateAction<FilterState>>;
  sortKey: SortKey;
  setSortKey: Dispatch<SetStateAction<SortKey>>;
  descending: boolean;
  setDescending: Dispatch<SetStateAction<boolean>>;
  page: number;
  setPage: Dispatch<SetStateAction<number>>;
  selectedId: string | null;
  setSelectedId: Dispatch<SetStateAction<string | null>>;
  history: HistoryEntry[];
  recordHistory: (itemId: string, action: HistoryAction) => void;
  deleteHistory: (keys: string[]) => void;
  clearHistory: () => void;
  refreshHistory: () => void;
  messages: ChatMessage[];
  setMessages: Dispatch<SetStateAction<ChatMessage[]>>;
  notice: string | null;
  flash: (message: string) => void;
  backendStatus: BackendStatus;
  backendVersion: string;
  catalogRows: ScoredCatalogItem[];
  catalogTotal: number;
  catalogMetrics: LibraryMetrics;
  catalogLoading: boolean;
  catalogWarnings: string[];
  recall: RecallState;
  meta: MetaOptions;
  pageSize: number;
  refreshLibrary: () => void;
};

// Used only when the backend is unavailable and config.py cannot be queried.
const FALLBACK_PAGE_SIZE = 500;
const emptyMetrics: LibraryMetrics = { items: 0, artists: 0, tags: 0, titleWords: 0 };
const emptyMeta: MetaOptions = { pageSize: FALLBACK_PAGE_SIZE, tags: [], artists: [], titleWords: [], languages: [], metrics: emptyMetrics };
const initialFilters: FilterState = {
  keyword: "", keywordRelevance: false, semanticQuery: "", coverQuery: "", coverFileName: "", coverMatches: {},
  weights: { tag: 1, artist: 1, title: 1, history: 1 }, minScore: 0, blockedTags: [],
  tagWeights: {}, artistWeights: {}, titleWeights: {},
};
const AppStateContext = createContext<AppStateValue | null>(null);

function formatNow() {
  return new Intl.DateTimeFormat("zh-CN", { year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false }).format(new Date()).replaceAll("/", "-");
}

function mockResults(filters: FilterState, history: HistoryEntry[], sortKey: SortKey, descending: boolean) {
  const keyword = filters.keyword.trim().toLowerCase();
  const historyIds = new Set(history.map((entry) => entry.itemId));
  const globalMultiplier = filters.weights.tag * 0.34 + filters.weights.artist * 0.24 + filters.weights.title * 0.22 + filters.weights.history * 0.2;
  const scored = catalogItems
    .filter((item) => !item.tags.some((tag) => filters.blockedTags.includes(tag)))
    .filter((item) => !keyword || [item.id, item.titleZh, item.title, item.artist, item.circle, item.language, ...item.tags].join(" ").toLowerCase().includes(keyword))
    .map((item) => {
      const tagBonus = item.tags.reduce((total, tag) => total + ((filters.tagWeights[tag] ?? 1) - 1) * 36, 0);
      const artistBonus = ((filters.artistWeights[item.artist] ?? 1) - 1) * 26;
      const titleBonus = Object.entries(filters.titleWeights).reduce((total, [word, weight]) => total + (item.titleZh.includes(word) || item.title.toLowerCase().includes(word.toLowerCase()) ? (weight - 1) * 24 : 0), 0);
      const historyBonus = historyIds.has(item.id) ? filters.weights.history * 18 : 0;
      return { ...item, score: Math.max(0, Math.round(item.baseScore * globalMultiplier + tagBonus + artistBonus + titleBonus + historyBonus)) };
    })
    .filter((item) => item.score >= filters.minScore);
  const direction = descending ? -1 : 1;
  return scored.sort((a, b) => {
    let result = 0;
    if (sortKey === "score") result = a.score - b.score;
    else if (sortKey === "keyword") result = a.keywordRelevance - b.keywordRelevance;
    else if (sortKey === "semantic") result = a.aiRelevance - b.aiRelevance;
    else if (sortKey === "cover") result = a.coverRelevance - b.coverRelevance;
    else if (sortKey === "date") result = a.uploadedAt.localeCompare(b.uploadedAt);
    else if (sortKey === "pages") result = a.pages - b.pages;
    else if (sortKey === "titleZh") result = a.titleZh.localeCompare(b.titleZh, "zh-CN");
    else if (sortKey === "title") result = a.title.localeCompare(b.title);
    else if (sortKey === "artist") result = a.artist.localeCompare(b.artist);
    else if (sortKey === "circle") result = a.circle.localeCompare(b.circle);
    else if (sortKey === "tags") result = a.tags.join(",").localeCompare(b.tags.join(","));
    else if (sortKey === "language") result = a.language.localeCompare(b.language);
    else if (sortKey === "localPath") result = a.localPath.localeCompare(b.localPath);
    else result = a.id.localeCompare(b.id);
    return result * direction;
  });
}

export function AppStateProvider({ children }: { children: ReactNode }) {
  const [filters, setFilters] = useState(initialFilters);
  const [sortKey, setSortKey] = useState<SortKey>("score");
  const [descending, setDescending] = useState(true);
  const [page, setPage] = useState(0);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [notice, setNotice] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<BackendStatus>("checking");
  const [backendVersion, setBackendVersion] = useState("");
  const [catalogRows, setCatalogRows] = useState<ScoredCatalogItem[]>([]);
  const [catalogTotal, setCatalogTotal] = useState(0);
  const [catalogMetrics, setCatalogMetrics] = useState(emptyMetrics);
  const [catalogLoading, setCatalogLoading] = useState(false);
  const [catalogWarnings, setCatalogWarnings] = useState<string[]>([]);
  const [recall, setRecall] = useState<RecallState>({ mode: "none", candidateCount: 0, usedFulltext: false });
  const [meta, setMeta] = useState<MetaOptions>(emptyMeta);
  const [refreshToken, setRefreshToken] = useState(0);

  const flash = useCallback((message: string) => {
    setNotice(message);
    window.setTimeout(() => setNotice(null), 2600);
  }, []);

  const refreshHistory = useCallback(() => {
    if (backendStatus !== "online") return;
    void getHistory().then((payload) => setHistory(payload.entries)).catch((error: Error) => flash(`历史记录读取失败：${error.message}`));
  }, [backendStatus, flash]);

  useEffect(() => {
    const controller = new AbortController();
    void getHealth(controller.signal)
      .then((health) => {
        setBackendVersion(health.version);
        setBackendStatus("online");
        void Promise.all([getMeta(controller.signal), getHistory(controller.signal)])
          .then(([metaPayload, historyPayload]) => {
            setMeta(metaPayload);
            setHistory(historyPayload.entries);
          })
          .catch((error: Error) => {
            if (error.name !== "AbortError") setCatalogWarnings([`选项元数据读取失败：${error.message}`]);
          });
      })
      .catch(() => setBackendStatus("offline"));
    return () => controller.abort();
  }, []);

  const filtersSignature = JSON.stringify(filters);
  const pageSize = meta.pageSize;
  useEffect(() => {
    if (backendStatus === "checking") return;
    if (backendStatus === "offline") {
      const allRows = mockResults(filters, history, sortKey, descending);
      setCatalogRows(allRows.slice(page * pageSize, (page + 1) * pageSize));
      setCatalogTotal(allRows.length);
      setCatalogMetrics({ items: catalogItems.length, artists: new Set(catalogItems.map((item) => item.artist)).size, tags: new Set(catalogItems.flatMap((item) => item.tags)).size, titleWords: titleWordOptions.length });
      setCatalogWarnings(["后端不可用，当前显示内置演示数据。"]);
      setRecall({ mode: "DEMO", candidateCount: allRows.length, usedFulltext: false });
      return;
    }
    const controller = new AbortController();
    setCatalogLoading(true);
    const timer = window.setTimeout(() => {
      void queryLibrary(filters, sortKey, descending, page, pageSize, controller.signal)
        .then((payload) => {
          setCatalogRows(payload.items);
          setCatalogTotal(payload.total);
          setCatalogMetrics(payload.metrics);
          setCatalogWarnings(payload.warnings);
          setRecall(payload.recall);
        })
        .catch((error: Error) => {
          if (error.name !== "AbortError") setCatalogWarnings([`库存请求失败：${error.message}`]);
        })
        .finally(() => setCatalogLoading(false));
    }, 220);
    return () => { controller.abort(); window.clearTimeout(timer); };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [backendStatus, filtersSignature, history, sortKey, descending, page, pageSize, refreshToken]);

  const recordHistory = useCallback((itemId: string, action: HistoryAction) => {
    if (backendStatus === "online") {
      void addHistory(itemId, action).then((payload) => { setHistory(payload.entries); flash(`已记录：${action}`); }).catch((error: Error) => flash(`记录失败：${error.message}`));
      return;
    }
    setHistory((current) => [{ key: `${Date.now()}-${itemId}`, openedAt: formatNow(), action, itemId }, ...current].slice(0, 50));
    flash(`演示记录：${action}`);
  }, [backendStatus, flash]);

  const deleteHistory = useCallback((keys: string[]) => {
    if (backendStatus === "online") void removeHistory(keys).then((payload) => setHistory(payload.entries)).catch((error: Error) => flash(`删除失败：${error.message}`));
    else setHistory((current) => current.filter((entry) => !keys.includes(entry.key)));
  }, [backendStatus, flash]);

  const clearHistory = useCallback(() => {
    if (backendStatus === "online") void removeAllHistory().then((payload) => setHistory(payload.entries)).catch((error: Error) => flash(`清空失败：${error.message}`));
    else setHistory([]);
  }, [backendStatus, flash]);

  const value = useMemo<AppStateValue>(() => ({
    filters, setFilters, sortKey, setSortKey, descending, setDescending, page, setPage,
    selectedId, setSelectedId, history, recordHistory, deleteHistory, clearHistory, refreshHistory,
    messages, setMessages, notice, flash, backendStatus, backendVersion, catalogRows, catalogTotal,
    catalogMetrics, catalogLoading, catalogWarnings, recall, meta, pageSize,
    refreshLibrary: () => setRefreshToken((value) => value + 1),
  }), [filters, sortKey, descending, page, selectedId, history, recordHistory, deleteHistory, clearHistory, refreshHistory, messages, notice, flash, backendStatus, backendVersion, catalogRows, catalogTotal, catalogMetrics, catalogLoading, catalogWarnings, recall, meta, pageSize]);

  return <AppStateContext.Provider value={value}>{children}</AppStateContext.Provider>;
}

export function useAppState() {
  const value = useContext(AppStateContext);
  if (!value) throw new Error("useAppState must be used inside AppStateProvider");
  return value;
}

export const titleWordOptions = ["雨夜", "少女", "森林", "猫", "夏日", "旧书", "灯塔", "机械", "星屑", "邮局"];
export function useCatalogResults() { return useAppState().catalogRows; }
export function findCatalogItem(itemId: string | null | undefined) { return catalogItems.find((item) => item.id === itemId); }
