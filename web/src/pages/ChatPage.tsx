import { Bot, ChevronDown, Send, SlidersHorizontal, UserRound } from "lucide-react";
import { useEffect, useRef, useState, type FormEvent } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import { Link } from "react-router-dom";
import remarkGfm from "remark-gfm";
import { streamChat, type ChatStreamEvent } from "../api/client";
import { EditorialCover } from "../components/CatalogTable";
import { LLMConnectionSettings, type ChatApiMode } from "../components/LLMConnectionSettings";
import { useAppState, useCatalogResults, type ChatMessage, type ScoredCatalogItem } from "../state/AppState";

const markdownComponents: Components = {
  a: ({ href, children }) => {
    const external = Boolean(href && /^https?:\/\//i.test(href));
    return <a href={href} target={external ? "_blank" : undefined} rel={external ? "noopener noreferrer" : undefined}>{children}</a>;
  },
};

function MessageMarkdown({ content }: { content: string }) {
  return (
    <div className="message-markdown">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents} skipHtml>{content}</ReactMarkdown>
    </div>
  );
}

function sampleWithoutReplacement<T>(items: readonly T[], requestedCount: number) {
  const count = Math.min(items.length, Math.max(0, Math.trunc(requestedCount)));
  const pool = [...items];
  for (let index = 0; index < count; index += 1) {
    const randomIndex = index + Math.floor(Math.random() * (pool.length - index));
    [pool[index], pool[randomIndex]] = [pool[randomIndex], pool[index]];
  }
  return pool.slice(0, count);
}

function snapshotItem(item: ScoredCatalogItem): ScoredCatalogItem {
  return { ...item, tags: [...item.tags] };
}

function parseLegacyChunk(rawResponse: string, exposeThinking: boolean) {
  const normalized = rawResponse.replace("Thinking Process:", "<think>");
  const thinkStart = normalized.indexOf("<think>");
  if (thinkStart < 0) return { content: normalized, thinking: "", thinkingStreaming: false };

  const thinkEnd = normalized.indexOf("</think>", thinkStart + 7);
  const prefix = normalized.slice(0, thinkStart);
  if (thinkEnd < 0) {
    return {
      content: prefix,
      thinking: exposeThinking ? normalized.slice(thinkStart + 7).trimStart() : "",
      thinkingStreaming: exposeThinking,
    };
  }
  return {
    content: `${prefix}${normalized.slice(thinkEnd + 8)}`.trimStart(),
    thinking: exposeThinking ? normalized.slice(thinkStart + 7, thinkEnd).trim() : "",
    thinkingStreaming: false,
  };
}

function ChatTurn({ message, index }: { message: ChatMessage; index: number }) {
  const assistant = message.role === "assistant";
  const [thinkingOpen, setThinkingOpen] = useState(Boolean(message.thinkingStreaming));
  const wasThinking = useRef(Boolean(message.thinkingStreaming));

  useEffect(() => {
    if (message.thinkingStreaming) {
      wasThinking.current = true;
      setThinkingOpen(true);
    } else if (wasThinking.current) {
      wasThinking.current = false;
      setThinkingOpen(false);
    }
  }, [message.thinkingStreaming]);

  return (
    <article className={`interview-turn interview-${message.role}`}>
      <div className="turn-index mono">{String(index + 1).padStart(2, "0")} / {assistant ? "ANSWER" : "QUESTION"}</div>
      <div className="turn-role">{assistant ? <Bot size={18} /> : <UserRound size={18} />}{assistant ? "ARCHIVE ASSISTANT" : "YOU"}</div>
      {assistant && message.requestMeta && <p className="turn-request-meta mono">{message.requestMeta}</p>}
      {assistant && message.thinking && (
        <details
          className="thinking-note"
          open={thinkingOpen}
          onToggle={(event) => setThinkingOpen(event.currentTarget.open)}
        >
          <summary>
            <span>LLM 思考过程</span>
            <span className="thinking-state mono">{message.thinkingStreaming ? "正在思考…" : "已完成"}</span>
          </summary>
          <MessageMarkdown content={message.thinking} />
        </details>
      )}
      {message.content ? <MessageMarkdown content={message.content} /> : assistant && <div className="message-markdown message-pending"><span className="type-cursor" aria-label="生成中" /></div>}
      {assistant && message.injectedItems && message.injectedItems.length > 0 && (
        <details className="context-injection">
          <summary>
            <span>本轮随机注入条目</span>
            <span className="mono">{message.injectedItems.length}</span>
            <ChevronDown size={15} aria-hidden="true" />
          </summary>
          <ol className="citation-list" aria-label="本轮随机注入的库存条目">
            {message.injectedItems.map((item, itemIndex) => (
              <li key={item.id}>
                <span className="citation-number mono">[{itemIndex + 1}]</span>
                <EditorialCover item={item} />
                <div><span className="mono">{item.id}</span><Link to={`/detail/${item.id}`} viewTransition>{item.titleZh || item.title}</Link><small>{item.artist || "—"}</small></div>
              </li>
            ))}
          </ol>
        </details>
      )}
    </article>
  );
}

export function ChatPage() {
  const { messages, setMessages, backendStatus, flash } = useAppState();
  const results = useCatalogResults();
  const [apiMode, setApiMode] = useState<ChatApiMode>("本地 (LM Studio)");
  const [contextCount, setContextCount] = useState(10);
  const [deepThinking, setDeepThinking] = useState(true);
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(4096);
  const [prompt, setPrompt] = useState("");
  const [streaming, setStreaming] = useState(false);
  const streamController = useRef<AbortController | null>(null);
  const streamFrame = useRef<number | null>(null);

  useEffect(() => () => {
    streamController.current?.abort();
    if (streamFrame.current !== null) window.cancelAnimationFrame(streamFrame.current);
  }, []);

  useEffect(() => {
    if (results.length > 0 && contextCount > results.length) setContextCount(results.length);
  }, [contextCount, results.length]);

  const effectiveContextCount = results.length ? Math.min(contextCount, results.length) : 0;

  const send = (event: FormEvent) => {
    event.preventDefault();
    const question = prompt.trim();
    if (!question || streaming) return;

    const contextItems = sampleWithoutReplacement(results, effectiveContextCount);
    const injectedItems = contextItems.map(snapshotItem);
    const contextIds = injectedItems.map((item) => item.id);
    const itemSnapshots = new Map(injectedItems.map((item) => [item.id, item]));
    const requestMeta = `接口模式：${apiMode}；Temperature：${temperature.toFixed(1)}；最大输出：${maxTokens}；深度思考：${deepThinking ? "开启" : "关闭"}；从库存当前页 ${results.length} 条中随机注入 ${injectedItems.length} 条作为 RAG 上下文。`;
    const timestamp = Date.now();
    const assistantId = `assistant-${timestamp}`;
    setMessages((current) => [
      ...current,
      { id: `user-${timestamp}`, role: "user", content: question },
      {
        id: assistantId,
        role: "assistant",
        content: "",
        requestMeta,
        thinkingStreaming: false,
        injectedItems,
      },
    ]);
    setPrompt("");
    setStreaming(true);

    if (backendStatus !== "online") {
      const answer = injectedItems.length
        ? `当前处于离线演示模式。我从本页随机选出的 ${injectedItems.length} 条样本中建议先查看《${injectedItems[0].titleZh}》。启动后端后，这里会改用配置的 LLM 流式回答。`
        : "当前处于离线演示模式且库存当前页为空。";
      setMessages((current) => current.map((message) => message.id === assistantId ? { ...message, content: answer, thinkingStreaming: false } : message));
      setStreaming(false);
      return;
    }

    const controller = new AbortController();
    streamController.current = controller;
    let pendingContent = "";
    let pendingThinking = "";
    let pendingThinkingStreaming: boolean | undefined;
    let pendingInjectedItems: ScoredCatalogItem[] | undefined;
    let pendingLegacy: ReturnType<typeof parseLegacyChunk> | undefined;
    let legacyResponse = "";
    let receivedAnswerContent = false;

    const flushPending = () => {
      if (streamFrame.current !== null) window.cancelAnimationFrame(streamFrame.current);
      streamFrame.current = null;
      const content = pendingContent;
      const thinking = pendingThinking;
      const thinkingStreaming = pendingThinkingStreaming;
      const nextInjectedItems = pendingInjectedItems;
      const legacy = pendingLegacy;
      pendingContent = "";
      pendingThinking = "";
      pendingThinkingStreaming = undefined;
      pendingInjectedItems = undefined;
      pendingLegacy = undefined;
      if (!content && !thinking && thinkingStreaming === undefined && !nextInjectedItems && !legacy) return;
      setMessages((current) => current.map((message) => {
        if (message.id !== assistantId) return message;
        const next = { ...message };
        if (legacy) {
          next.content = legacy.content;
          next.thinking = legacy.thinking || undefined;
          next.thinkingStreaming = legacy.thinkingStreaming;
        }
        if (content) next.content = `${next.content}${content}`;
        if (thinking) next.thinking = `${next.thinking ?? ""}${thinking}`;
        if (thinkingStreaming !== undefined) next.thinkingStreaming = thinkingStreaming;
        if (nextInjectedItems) next.injectedItems = nextInjectedItems;
        return next;
      }));
    };

    const scheduleFlush = () => {
      if (streamFrame.current === null) streamFrame.current = window.requestAnimationFrame(flushPending);
    };

    const handleStreamEvent = (streamEvent: ChatStreamEvent) => {
      if (streamEvent.type === "meta" && streamEvent.contextIds) {
        const resolved = streamEvent.contextIds.map((id) => itemSnapshots.get(id)).filter((item): item is ScoredCatalogItem => Boolean(item));
        if (resolved.length === streamEvent.contextIds.length) pendingInjectedItems = resolved;
      } else if (streamEvent.type === "reasoning" && streamEvent.content) {
        if (deepThinking) {
          pendingThinking += streamEvent.content;
          pendingThinkingStreaming = true;
        }
      } else if (streamEvent.type === "reasoning_done") {
        pendingThinkingStreaming = false;
      } else if (streamEvent.type === "content" && streamEvent.content) {
        pendingContent += streamEvent.content;
        receivedAnswerContent = true;
        pendingThinkingStreaming = false;
      } else if (streamEvent.type === "chunk" && streamEvent.content) {
        legacyResponse += streamEvent.content;
        pendingLegacy = parseLegacyChunk(legacyResponse, deepThinking);
        receivedAnswerContent = Boolean(pendingLegacy.content);
      } else if (streamEvent.type === "error") {
        const errorMessage = streamEvent.message || streamEvent.content || "上游接口返回未知错误";
        pendingContent += `${receivedAnswerContent ? "\n\n" : ""}连接失败：${errorMessage}`;
        pendingThinkingStreaming = false;
      } else if (streamEvent.type === "done") {
        pendingThinkingStreaming = false;
        flushPending();
        return;
      }
      scheduleFlush();
    };

    void streamChat({
      query: question,
      apiMode,
      temperature,
      maxTokens,
      contextIds,
      contextCount: injectedItems.length,
      deepThinking,
    }, handleStreamEvent, controller.signal).catch((error: Error) => {
      if (error.name !== "AbortError") {
        pendingContent += `${receivedAnswerContent ? "\n\n" : ""}连接失败：${error.message}`;
        pendingThinkingStreaming = false;
        flushPending();
      }
    }).finally(() => {
      pendingThinkingStreaming = false;
      flushPending();
      streamController.current = null;
      setStreaming(false);
    });
  };

  const olderMessages = messages.slice(0, -2);
  const latestMessages = messages.slice(-2);

  return (
    <div className="chat-layout">
      <aside className="chat-settings" aria-label="LLM 助手设置">
        <span className="section-code">MARGIN NOTES / MODEL</span>
        <h2><SlidersHorizontal size={18} />访谈设置</h2>
        <fieldset>
          <legend>API 来源</legend>
          {(["本地 (LM Studio)", "线上 API"] as ChatApiMode[]).map((mode) => (
            <label className="editorial-radio" key={mode}><input type="radio" name="api-mode" checked={apiMode === mode} onChange={() => setApiMode(mode)} /><span />{mode}</label>
          ))}
        </fieldset>
        <label className="thinking-switch" htmlFor="deep-thinking">
          <span><b>深度思考模式</b><small id="deep-thinking-help">请求并流式展示模型返回的独立思考过程。</small></span>
          <input id="deep-thinking" type="checkbox" role="switch" checked={deepThinking} aria-describedby="deep-thinking-help" onChange={(event) => setDeepThinking(event.target.checked)} />
          <i aria-hidden="true"><i /></i>
        </label>
        <LLMConnectionSettings apiMode={apiMode} backendStatus={backendStatus} flash={flash} />
        <label>随机注入条目数 <span className="mono">当前页上限 {results.length}</span><input className="mono" type="number" min="0" max={results.length} step="1" value={effectiveContextCount} disabled={!results.length} onChange={(event) => {
          const value = Number(event.target.value);
          if (Number.isFinite(value)) setContextCount(Math.min(results.length, Math.max(0, Math.trunc(value))));
        }} /></label>
        {!results.length && <p className="control-help">库存当前页没有数据，助手只能基于通用知识回答。</p>}
        <label>温度（Temp）<input className="mono" type="number" min="0" max="2" step="0.1" value={temperature} onChange={(event) => setTemperature(Number(event.target.value))} /></label>
        <label>最大输出（Tokens）<input className="mono" type="number" min="256" max="32768" step="512" value={maxTokens} onChange={(event) => setMaxTokens(Number(event.target.value))} /></label>
        <p className="footnote"><sup>1</sup> {backendStatus === "online" ? "回答由后端代理到已配置的 LLM，并以 SSE 流式返回。每轮会从库存当前页重新随机抽取上下文。" : "后端离线，当前仅显示本地演示回答。"}</p>
      </aside>

      <section className="chat-page">
        <header className="page-intro">
          <span className="section-code">SECTION 03 / INTERVIEW</span>
          <h2>何不试试<br />LLM 呢？</h2>
          <p>从库存当前页随机抽取访谈材料。回答下方可以展开本轮实际注入的条目并回到漫画详情。</p>
        </header>

        {olderMessages.length > 0 && (
          <details className="conversation-history">
            <summary>查看历史对话记录 <span className="mono">{olderMessages.length}</span></summary>
            {olderMessages.map((message, index) => <ChatTurn message={message} index={index} key={message.id} />)}
          </details>
        )}

        <section className="latest-conversation" aria-live="polite">
          {latestMessages.map((message, index) => <ChatTurn message={message} index={messages.length - latestMessages.length + index} key={message.id} />)}
        </section>

        <form className="chat-composer" onSubmit={send}>
          <label htmlFor="chat-prompt">问问 LLM 关于这些收藏的事</label>
          <div>
            <textarea id="chat-prompt" rows={3} value={prompt} aria-describedby="chat-markdown-help" onChange={(event) => setPrompt(event.target.value)} placeholder="例如：从当前结果里找三部适合周末阅读的长篇……" />
            <button type="submit" disabled={!prompt.trim() || streaming}><Send size={16} />{streaming ? "生成中" : "发送问题"}</button>
          </div>
          <small id="chat-markdown-help" className="chat-markdown-help mono">支持 Markdown 与 GFM：标题、列表、链接、引用、代码和表格。</small>
        </form>
      </section>
    </div>
  );
}
