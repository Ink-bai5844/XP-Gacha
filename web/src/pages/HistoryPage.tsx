import { ExternalLink, FolderOpen, RefreshCw, Trash2, X } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";
import { openLocal } from "../api/client";
import { findCatalogItem, titleWordOptions, useAppState } from "../state/AppState";

export function HistoryPage() {
  const { history, clearHistory, deleteHistory, recordHistory, refreshHistory, backendStatus, flash } = useAppState();
  const [selected, setSelected] = useState<string[]>([]);
  const [confirmClear, setConfirmClear] = useState(false);

  const removeSelected = () => {
    deleteHistory(selected);
    flash(`已删除 ${selected.length} 条历史记录`);
    setSelected([]);
  };

  return (
    <div className="history-page">
      <header className="page-intro page-intro-split">
        <div><span className="section-code">SECTION 04 / REFERENCES</span><h2>历史记录</h2></div>
        <p>缓存最近 50 次来源链接或本地打开记录，当前已保存 <strong className="mono">{history.length}</strong> 条。最近记录会参与历史偏好加权。</p>
      </header>

      <div className="history-toolbar">
        <button type="button" onClick={() => { refreshHistory(); flash(`正在刷新历史记录 · 当前 ${history.length} 条`); }}><RefreshCw size={14} />刷新记录</button>
        {!confirmClear ? (
          <button type="button" className="ink-button" disabled={!history.length} onClick={() => setConfirmClear(true)}><Trash2 size={14} />清空记录</button>
        ) : (
          <div className="confirm-inline" role="alert">
            <span>确认清空全部 {history.length} 条？</span>
            <button type="button" onClick={() => { clearHistory(); setSelected([]); setConfirmClear(false); flash("已清空历史偏好记录"); }}>确认清空</button>
            <button type="button" onClick={() => setConfirmClear(false)} aria-label="取消清空"><X size={14} /></button>
          </div>
        )}
      </div>

      {!history.length ? (
        <div className="history-empty"><span className="mono">REFERENCES / 000</span><h3>暂时还没有保存的历史记录</h3><p>从漫画详情打开本地目录或网络来源后，记录会出现在这里。</p><Link to="/">返回库存目录</Link></div>
      ) : (
        <>
          <ol className="reference-list">
            {history.map((entry, index) => {
              const item = findCatalogItem(entry.itemId);
              const title = entry.title || item?.titleZh || entry.itemId;
              const author = entry.author || item?.artist || "—";
              const tags = entry.tags?.length ? entry.tags : item?.tags ?? [];
              const words = entry.titleWords?.length ? entry.titleWords : titleWordOptions.filter((word) => title.includes(word));
              const localPath = entry.localPath || item?.localPath || "本地目录不存在";
              return (
                <li key={entry.key}>
                  <label className="history-select">
                    <input type="checkbox" checked={selected.includes(entry.key)} onChange={(event) => setSelected((current) => event.target.checked ? [...current, entry.key] : current.filter((key) => key !== entry.key))} />
                    <span aria-hidden="true" />
                    <span className="sr-only">选择历史记录 {title}</span>
                  </label>
                  <span className="reference-index mono">[{String(index + 1).padStart(2, "0")}]</span>
                  <div className="reference-main">
                    <span className="reference-time mono">{entry.openedAt} · {entry.action}</span>
                    <h3><Link to={`/detail/${entry.itemId}`} viewTransition>{title}</Link></h3>
                    <p>{author} · {entry.itemId} · {tags.join("，")}</p>
                    <small>标题词：{words.join(" | ") || "—"}</small>
                  </div>
                  <div className="reference-actions">
                    <button type="button" disabled={localPath === "本地目录不存在"} onClick={() => {
                      if (backendStatus === "online") void openLocal(entry.itemId).then(refreshHistory).catch((error: Error) => flash(error.message));
                      else recordHistory(entry.itemId, "打开本地目录");
                    }}><FolderOpen size={13} />打开本地</button>
                    <a href={entry.link ? `/api/track/${encodeURIComponent(entry.itemId)}` : `https://example.com/gallery/${entry.itemId}`} target="_blank" rel="noreferrer" onClick={() => { if (!entry.link) recordHistory(entry.itemId, "打开网络来源"); }}><ExternalLink size={13} />打开链接</a>
                    <button type="button" onClick={() => { deleteHistory([entry.key]); flash("已删除 1 条历史记录"); }}><Trash2 size={13} />删除</button>
                  </div>
                </li>
              );
            })}
          </ol>
          <button className="delete-selected" type="button" disabled={!selected.length} onClick={removeSelected}><Trash2 size={14} />删除选中的 {selected.length} 条记录</button>
        </>
      )}

    </div>
  );
}
