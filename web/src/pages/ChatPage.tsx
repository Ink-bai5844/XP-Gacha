import { Bot, Send, SlidersHorizontal, UserRound } from "lucide-react";
import { useEffect, useRef, useState, type FormEvent } from "react";
import { Link } from "react-router-dom";
import { streamChat } from "../api/client";
import { EditorialCover } from "../components/CatalogTable";
import { findCatalogItem, useAppState, useCatalogResults, type ChatMessage } from "../state/AppState";

function ChatTurn({ message, index, inventory }: { message: ChatMessage; index: number; inventory: ReturnType<typeof useCatalogResults> }) {
  const assistant = message.role === "assistant";
  return (
    <article className={`interview-turn interview-${message.role}`}>
      <div className="turn-index mono">{String(index + 1).padStart(2, "0")} / {assistant ? "ANSWER" : "QUESTION"}</div>
      <div className="turn-role">{assistant ? <Bot size={18} /> : <UserRound size={18} />}{assistant ? "ARCHIVE ASSISTANT" : "MOBai"}</div>
      {assistant ? <p className="assistant-copy">{message.content || <span className="type-cursor" aria-label="生成中" />}</p> : <h3>{message.content}</h3>}
      {assistant && message.thinking && (
        <details className="thinking-note">
          <summary>LLM 思考过程</summary>
          <p>{message.thinking}</p>
        </details>
      )}
      {assistant && message.citations && message.citations.length > 0 && (
        <ol className="citation-list" aria-label="库存引用">
          {message.citations.map((id, citationIndex) => {
            const item = inventory.find((candidate) => candidate.id === id) ?? findCatalogItem(id);
            if (!item) return null;
            return (
              <li key={id}>
                <span className="citation-number mono">[{citationIndex + 1}]</span>
                <EditorialCover item={item} />
                <div><span className="mono">{item.id}</span><Link to={`/detail/${item.id}`} viewTransition>{item.titleZh}</Link><small>{item.artist}</small></div>
              </li>
            );
          })}
        </ol>
      )}
    </article>
  );
}

export function ChatPage() {
  const { messages, setMessages, backendStatus } = useAppState();
  const results = useCatalogResults();
  const [apiMode, setApiMode] = useState("本地 (LM Studio)");
  const [contextCount, setContextCount] = useState(Math.min(10, results.length));
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(4096);
  const [prompt, setPrompt] = useState("");
  const [streaming, setStreaming] = useState(false);
  const streamController = useRef<AbortController | null>(null);

  useEffect(() => () => {
    streamController.current?.abort();
  }, []);

  useEffect(() => {
    if (contextCount > results.length) setContextCount(results.length);
  }, [contextCount, results.length]);

  const send = (event: FormEvent) => {
    event.preventDefault();
    const question = prompt.trim();
    if (!question || streaming) return;
    const contextItems = results.slice(0, Math.max(0, contextCount));
    const citations = contextItems.map((item) => item.id);
    const assistantId = `assistant-${Date.now()}`;
    setMessages((current) => [
      ...current,
      { id: `user-${Date.now()}`, role: "user", content: question },
      {
        id: assistantId,
        role: "assistant",
        content: "",
        thinking: `接口模式：${apiMode}；Temperature：${temperature.toFixed(1)}；最大输出：${maxTokens}；按当前排序抽取前 ${contextItems.length} 条作为 RAG 上下文。`,
        citations,
      },
    ]);
    setPrompt("");
    setStreaming(true);
    if (backendStatus !== "online") {
      const answer = contextItems.length
        ? `当前处于离线演示模式。我从本页 ${contextItems.length} 条样本中建议先查看《${contextItems[0].titleZh}》。启动后端后，这里会改用配置的 LLM 流式回答。`
        : "当前处于离线演示模式且筛选结果为空。";
      setMessages((current) => current.map((message) => message.id === assistantId ? { ...message, content: answer } : message));
      setStreaming(false);
      return;
    }
    const controller = new AbortController();
    streamController.current = controller;
    let rawResponse = "";
    void streamChat({
      query: question,
      apiMode,
      temperature,
      maxTokens,
      contextIds: citations,
      contextCount: contextItems.length,
    }, (streamEvent) => {
      if (streamEvent.type === "chunk" && streamEvent.content) {
        rawResponse += streamEvent.content;
        const normalized = rawResponse.replace("Thinking Process:", "<think>");
        const thinkStart = normalized.indexOf("<think>");
        const thinkEnd = normalized.indexOf("</think>");
        const thinking = thinkStart >= 0
          ? normalized.slice(thinkStart + 7, thinkEnd >= 0 ? thinkEnd : undefined).trim()
          : undefined;
        const content = thinkEnd >= 0 ? normalized.slice(thinkEnd + 8).trimStart() : (thinkStart >= 0 ? "" : normalized);
        setMessages((current) => current.map((message) => message.id === assistantId ? { ...message, content, thinking: thinking || message.thinking } : message));
      }
    }, controller.signal).catch((error: Error) => {
      if (error.name !== "AbortError") setMessages((current) => current.map((message) => message.id === assistantId ? { ...message, content: `${message.content}\n\n连接失败：${error.message}` } : message));
    }).finally(() => {
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
          {["本地 (LM Studio)", "线上 API"].map((mode) => (
            <label className="editorial-radio" key={mode}><input type="radio" name="api-mode" checked={apiMode === mode} onChange={() => setApiMode(mode)} /><span />{mode}</label>
          ))}
        </fieldset>
        <label>随机注入条目数 <span className="mono">上限 {results.length}</span><input className="mono" type="number" min="0" max={results.length} value={contextCount} disabled={!results.length} onChange={(event) => setContextCount(Number(event.target.value))} /></label>
        {!results.length && <p className="control-help">当前页面没有库存数据，助手只能基于通用知识回答。</p>}
        <label>温度（Temp）<input className="mono" type="number" min="0" max="2" step="0.1" value={temperature} onChange={(event) => setTemperature(Number(event.target.value))} /></label>
        <label>最大输出（Tokens）<input className="mono" type="number" min="256" max="32768" step="512" value={maxTokens} onChange={(event) => setMaxTokens(Number(event.target.value))} /></label>
        <p className="footnote"><sup>1</sup> {backendStatus === "online" ? "回答由后端代理到已配置的 LLM，并以 SSE 流式返回。" : "后端离线，当前仅显示本地演示回答。"}</p>
      </aside>

      <section className="chat-page">
        <header className="page-intro">
          <span className="section-code">SECTION 03 / INTERVIEW</span>
          <h2>何不试试<br />LLM 呢？</h2>
          <p>把当前筛选结果作为访谈材料。回答中的库存引用可以直接回到漫画详情。</p>
        </header>

        {olderMessages.length > 0 && (
          <details className="conversation-history">
            <summary>查看历史对话记录 <span className="mono">{olderMessages.length}</span></summary>
            {olderMessages.map((message, index) => <ChatTurn message={message} index={index} inventory={results} key={message.id} />)}
          </details>
        )}

        <section className="latest-conversation" aria-live="polite">
          {latestMessages.map((message, index) => <ChatTurn message={message} index={messages.length - latestMessages.length + index} inventory={results} key={message.id} />)}
        </section>

        <form className="chat-composer" onSubmit={send}>
          <label htmlFor="chat-prompt">问问 LLM 关于这些收藏的事</label>
          <div>
            <textarea id="chat-prompt" rows={3} value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder="例如：从当前结果里找三部适合周末阅读的长篇……" />
            <button type="submit" disabled={!prompt.trim() || streaming}><Send size={16} />{streaming ? "生成中" : "发送问题"}</button>
          </div>
        </form>
      </section>
    </div>
  );
}
