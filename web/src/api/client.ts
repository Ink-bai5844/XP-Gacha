import type { CatalogItem } from "../data/catalog";
import type { FilterState, HistoryEntry, SortKey } from "../state/AppState";

export type BackendStatus = "checking" | "online" | "offline";

export type LibraryMetrics = {
  items: number;
  artists: number;
  tags: number;
  titleWords: number;
};

export type LibraryResponse = {
  items: Array<CatalogItem & { score: number }>;
  total: number;
  page: number;
  pageSize: number;
  metrics: LibraryMetrics;
  recall: { mode: string; candidateCount: number; usedFulltext: boolean };
  warnings: string[];
};

export type MetaOptions = {
  pageSize: number;
  tags: string[];
  artists: string[];
  titleWords: string[];
  languages: string[];
  metrics: LibraryMetrics;
};

export type OptionKind = "tags" | "artists" | "titleWords";
export type OptionSearchResponse = { items: string[]; total: number; offset: number; hasMore: boolean };

export type SearchDependencyStatus = {
  kind: "model" | "vector";
  label: string;
  path: string;
  exists: boolean;
  ready: boolean;
  state: "ready" | "missing" | "incomplete";
  downloadUrl?: string;
  generatedLocally?: boolean;
};

export type SearchCapabilityStatus = {
  label: string;
  ready: boolean;
  idReady?: boolean;
  uploadReady?: boolean;
  missing: Array<"model" | "vector">;
  dependencies: { model: SearchDependencyStatus; vector: SearchDependencyStatus };
  setup: { section: string; scriptId: string; actionLabel: string };
};

export type SystemStatus = {
  database: { available: boolean; table_ready: boolean; row_count: number; error?: string };
  models: { semantic: boolean; clip: boolean };
  searchCapabilities: {
    semantic: SearchCapabilityStatus;
    cover: SearchCapabilityStatus;
  };
  counts: { csv: number; onlineCovers: number; localThumbnails: number; base64: number };
  caches: Array<{ name: string; path: string; exists: boolean; sizeKb: number }>;
  paths: { dataRoot: string; library: string; dictionaries: string };
};

export type LLMSettings = {
  local: { apiBase: string; model: string; apiKeyConfigured: boolean };
  online: { apiBase: string; model: string; apiKeyConfigured: boolean };
  persistence: {
    runtimeMode: "source" | "docker" | "portable" | string;
    fileName: string;
    writable: boolean;
    restartRequired: boolean;
  };
};

export type LLMSettingsUpdate = {
  localApiBase: string;
  localModel: string;
  localApiKey?: string;
  clearLocalApiKey: boolean;
  onlineApiBase: string;
  onlineModel: string;
  onlineApiKey?: string;
  clearOnlineApiKey: boolean;
};

async function readError(response: Response) {
  try {
    const payload = await response.json() as {
      detail?: string | Array<{ loc?: Array<string | number>; msg?: string }>;
    };
    if (Array.isArray(payload.detail)) {
      const messages = payload.detail
        .map((item) => {
          const field = item.loc?.at(-1);
          const message = item.msg?.replace(/^Value error,\s*/i, "");
          return [field, message].filter(Boolean).join("：");
        })
        .filter(Boolean);
      if (messages.length) return messages.join("；");
    }
    if (typeof payload.detail === "string") return payload.detail;
    return `${response.status} ${response.statusText}`;
  } catch {
    return `${response.status} ${response.statusText}`;
  }
}

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: init?.body instanceof FormData
      ? init.headers
      : { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) throw new Error(await readError(response));
  return response.json() as Promise<T>;
}

export function getHealth(signal?: AbortSignal) {
  return apiFetch<{
    status: string;
    version: string;
    database: { available: boolean; table_ready: boolean; row_count: number; error?: string };
  }>("/api/health", { signal });
}

export function queryLibrary(
  filters: FilterState,
  sort: SortKey,
  descending: boolean,
  page: number,
  pageSize: number,
  signal?: AbortSignal,
) {
  return apiFetch<LibraryResponse>("/api/library/query", {
    method: "POST",
    signal,
    body: JSON.stringify({ ...filters, sort, descending, page, pageSize }),
  });
}

export function getMeta(signal?: AbortSignal) {
  return apiFetch<MetaOptions>("/api/meta/options", { signal });
}

export function searchMetaOptions(kind: OptionKind, query = "", limit = 80, offset = 0, signal?: AbortSignal) {
  const params = new URLSearchParams({ kind, q: query, limit: String(limit), offset: String(offset) });
  return apiFetch<OptionSearchResponse>(`/api/meta/options/search?${params}`, { signal });
}

export function getGallery(itemId: string, signal?: AbortSignal) {
  return apiFetch<CatalogItem & { score: number; titleWords?: string[]; rawTags?: string[] }>(`/api/gallery/${encodeURIComponent(itemId)}`, { signal });
}

export function getHistory(signal?: AbortSignal) {
  return apiFetch<{ entries: HistoryEntry[] }>("/api/history", { signal });
}

export function addHistory(itemId: string, action: HistoryEntry["action"]) {
  return apiFetch<{ entries: HistoryEntry[] }>("/api/history", {
    method: "POST",
    body: JSON.stringify({ itemId, action }),
  });
}

export function removeHistory(keys: string[]) {
  return apiFetch<{ entries: HistoryEntry[] }>("/api/history", {
    method: "DELETE",
    body: JSON.stringify({ keys }),
  });
}

export function removeAllHistory() {
  return apiFetch<{ entries: HistoryEntry[] }>("/api/history/all", { method: "DELETE" });
}

export function openLocal(itemId: string) {
  return apiFetch<{ opened: boolean; path: string }>(`/api/gallery/${encodeURIComponent(itemId)}/open-local`, { method: "POST" });
}

export function refreshCovers(itemIds: string[]) {
  return apiFetch<{ queued: number }>("/api/covers/refresh", {
    method: "POST",
    body: JSON.stringify(itemIds),
  });
}

export async function searchCoverFile(file: File) {
  const form = new FormData();
  form.set("file", file);
  return apiFetch<{ results: Array<{ item_id: string; score: number }> }>("/api/search/cover", { method: "POST", body: form });
}

export type RankItem = { label: string; value: number };
export type ChartPayload = Record<string, {
  title: string;
  top_15: RankItem[];
  top_150: RankItem[];
  label_col: string;
  value_col: string;
}>;

export function getCharts(scope: "global" | "history", signal?: AbortSignal) {
  return apiFetch<ChartPayload>(`/api/charts/${scope}`, { signal });
}

export function getSystemStatus(signal?: AbortSignal) {
  return apiFetch<SystemStatus>("/api/system/status", { signal });
}

export function getLLMSettings(signal?: AbortSignal) {
  return apiFetch<LLMSettings>("/api/chat/settings", { signal });
}

export function saveLLMSettings(payload: LLMSettingsUpdate) {
  return apiFetch<LLMSettings>("/api/chat/settings", {
    method: "PUT",
    headers: { "X-XP-Gacha-Settings": "same-origin" },
    body: JSON.stringify(payload),
  });
}

export async function importBundle(file: File, mode: "upsert" | "replace", includeDictionaries = true) {
  const form = new FormData();
  form.set("file", file);
  form.set("mode", mode);
  form.set("include_dictionaries", String(includeDictionaries));
  return apiFetch<{ imported: number; total: number | null; csvFiles: number; dictionaries: string[] }>("/api/import/bundle", { method: "POST", body: form });
}

export function importProject(mode: "upsert" | "replace") {
  return apiFetch<{ imported: number; total: number; csvFiles: number }>("/api/import/project", {
    method: "POST",
    body: JSON.stringify({ mode, includeDictionaries: true }),
  });
}

export type JobResponse = {
  id: string;
  scriptId: string;
  status: "queued" | "running" | "cancelling" | "completed" | "failed" | "cancelled";
  lines: string[];
  lineCount: number;
  returnCode: number | null;
};

export function startJob(scriptId: string, parameters: Record<string, unknown>) {
  return apiFetch<JobResponse>("/api/jobs", {
    method: "POST",
    body: JSON.stringify({ scriptId, parameters }),
  });
}

export function getJob(jobId: string, after = 0) {
  return apiFetch<JobResponse>(`/api/jobs/${jobId}?after=${after}`);
}

export function cancelJob(jobId: string) {
  return apiFetch<JobResponse>(`/api/jobs/${jobId}/cancel`, { method: "POST" });
}

export type ChatStreamEvent = {
  type: "meta" | "reasoning" | "reasoning_done" | "content" | "error" | "done" | "chunk";
  content?: string;
  message?: string;
  contextIds?: string[];
};

export async function streamChat(
  payload: {
    query: string;
    apiMode: string;
    temperature: number;
    maxTokens: number;
    contextIds: string[];
    contextCount: number;
    deepThinking: boolean;
  },
  onEvent: (event: ChatStreamEvent) => void,
  signal?: AbortSignal,
) {
  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Accept": "text/event-stream", "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
  if (!response.ok) throw new Error(await readError(response));
  if (!response.body) throw new Error("浏览器不支持流式响应");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const dispatchBlock = (block: string) => {
    const data = block
      .split(/\r?\n/)
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trimStart())
      .join("\n");
    if (!data || data === "[DONE]") return;
    onEvent(JSON.parse(data) as ChatStreamEvent);
  };
  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop() ?? "";
    blocks.forEach(dispatchBlock);
    if (done) {
      if (buffer.trim()) dispatchBlock(buffer);
      break;
    }
  }
}
