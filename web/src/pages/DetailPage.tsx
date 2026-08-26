import { ArrowLeft, ExternalLink, FolderOpen } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { getGallery, openLocal } from "../api/client";
import { EditorialCover } from "../components/CatalogTable";
import { findCatalogItem, titleWordOptions, useAppState, useCatalogResults } from "../state/AppState";

export function DetailPage() {
  const { id } = useParams();
  const { selectedId, setSelectedId, recordHistory, refreshHistory, backendStatus } = useAppState();
  const results = useCatalogResults();
  const targetId = id === "demo" ? selectedId : id;
  const fallbackItem = results.find((candidate) => candidate.id === targetId) ?? findCatalogItem(targetId);
  const [remoteItem, setRemoteItem] = useState<(typeof fallbackItem & { score?: number; titleWords?: string[]; rawTags?: string[] }) | null>(null);
  const item = remoteItem ?? fallbackItem;
  const [actionNote, setActionNote] = useState("");

  useEffect(() => {
    if (!targetId || backendStatus !== "online") return;
    const controller = new AbortController();
    void getGallery(targetId, controller.signal).then(setRemoteItem).catch(() => setRemoteItem(null));
    return () => controller.abort();
  }, [backendStatus, targetId]);

  useEffect(() => {
    if (item) setSelectedId(item.id);
  }, [item, setSelectedId]);

  if (!item) {
    return (
      <section className="empty-detail">
        <span className="section-code">SECTION 02 / DETAIL</span>
        <h2>{selectedId ? "当前条目不在筛选结果中" : "还没有选中漫画"}</h2>
        <p>{selectedId ? "请调整筛选条件，或在库存列表重新选择一条记录。" : "先在库存目录中勾选一部漫画，完整元数据会显示在这里。"}</p>
        <Link to="/" viewTransition><ArrowLeft size={15} />返回库存目录</Link>
      </section>
    );
  }

  const scored = results.find((candidate) => candidate.id === item.id);
  const score = scored?.score ?? item.baseScore;
  const availableLocally = item.localPath !== "本地目录不存在";
  const titleWords = remoteItem?.titleWords ?? titleWordOptions.filter((word) => item.titleZh.includes(word));
  const rawTags = remoteItem?.rawTags?.length ? remoteItem.rawTags : item.tags;
  const metadata = [
    ["推荐评分", score],
    ["关键词相关度", item.keywordRelevance.toFixed(2)],
    ["封面相关度", item.coverRelevance.toFixed(2)],
    ["AI相关度", item.aiRelevance.toFixed(2)],
    ["ID", item.id],
    ["标题", item.title],
    ["标题译文", item.titleZh],
    ["作者", item.artist],
    ["团队", item.circle],
    ["上传日期", item.uploadedAt],
    ["语言", item.language],
    ["页数", item.pages],
    ["标签", rawTags.join(" | ") || "—"],
    ["解析后标签", item.tags.join(" | ")],
    ["标题特征词", titleWords.join(" | ") || "—"],
    ["本地目录", item.localPath],
    ["文件名", item.filename || `${item.id} ${item.titleZh}`],
    ["链接", item.link || `https://example.com/gallery/${item.id}`],
    ["搜索文本", [item.titleZh, item.title, item.artist, item.circle, ...item.tags].join(" ")],
  ];

  return (
    <article className="detail-page">
      <div className="detail-folio mono">SECTION 02 / DETAIL — {item.id}</div>
      <div className="detail-spread">
        <section className="detail-cover-page" aria-label="封面与入口">
          <div className="detail-cover-wipe">
            <EditorialCover item={item} large transitionName={`cover-${item.id}`} />
          </div>
          <div className="detail-actions">
            <a
              href={item.link ? `/api/track/${encodeURIComponent(item.id)}` : `https://example.com/gallery/${item.id}`}
              target="_blank"
              rel="noreferrer"
              onClick={() => { if (!item.link) recordHistory(item.id, "打开网络来源"); else window.setTimeout(refreshHistory, 500); setActionNote("已记录来源访问"); }}
            ><ExternalLink size={15} />打开网络来源</a>
            <button
              type="button"
              disabled={!availableLocally}
              onClick={() => {
                if (backendStatus === "online") void openLocal(item.id).then(({ opened, path }) => { setActionNote(`${opened ? "已打开" : "目录可用"} · ${path}`); refreshHistory(); }).catch((error: Error) => setActionNote(`打开失败 · ${error.message}`));
                else { recordHistory(item.id, "打开本地目录"); setActionNote(`演示记录 · ${item.localPath}`); }
              }}
            ><FolderOpen size={15} />打开本地目录</button>
          </div>
          {availableLocally
            ? <code className="detail-path">{item.localPath}</code>
            : <p className="detail-warning">本地目录不可用：未匹配</p>}
          {actionNote && <p className="action-note mono" role="status">{actionNote}</p>}
        </section>

        <section className="detail-copy-page">
          <Link className="detail-back" to="/" viewTransition><ArrowLeft size={14} />返回库存</Link>
          <span className="section-code">SELECTED MONOGRAPH / {item.id}</span>
          <h2 style={{ viewTransitionName: `title-${item.id}` }}>{item.titleZh}</h2>
          <p className="detail-original">{item.title}</p>
          <p className="detail-deck">{item.summary}</p>

          <div className="detail-metrics">
            <div><span>推荐分</span><strong className="mono signal-text">{score}</strong></div>
            <div><span>上传日期</span><strong className="mono">{item.uploadedAt}</strong></div>
            <div><span>语言</span><strong>{item.language}</strong></div>
            <div><span>页数</span><strong className="mono">{item.pages}</strong></div>
          </div>

          <dl className="detail-definition-list">
            {metadata.map(([field, value]) => (
              <div key={field}><dt>{field}</dt><dd className={field === "推荐评分" ? "mono signal-text" : ""}>{value}</dd></div>
            ))}
          </dl>
        </section>
      </div>
    </article>
  );
}
