import { NavLink, Route, Routes } from "react-router-dom";
import { lazy, Suspense } from "react";
import { useAppState } from "./state/AppState";

const CatalogPage = lazy(() => import("./pages/CatalogPage").then((module) => ({ default: module.CatalogPage })));
const DetailPage = lazy(() => import("./pages/DetailPage").then((module) => ({ default: module.DetailPage })));
const ChatPage = lazy(() => import("./pages/ChatPage").then((module) => ({ default: module.ChatPage })));
const HistoryPage = lazy(() => import("./pages/HistoryPage").then((module) => ({ default: module.HistoryPage })));
const ChartsPage = lazy(() => import("./pages/ChartsPage").then((module) => ({ default: module.ChartsPage })));
const AdminPage = lazy(() => import("./pages/AdminPage").then((module) => ({ default: module.AdminPage })));

export default function App() {
  const { selectedId, history, notice, backendStatus, backendVersion, catalogMetrics } = useAppState();
  const navItems = [
    ["/", "库存"],
    [`/detail/${selectedId ?? "demo"}`, "详情"],
    ["/chat", "助手"],
    ["/history", "历史"],
    ["/charts", "图表"],
    ["/admin", "附录"],
  ];
  return (
    <div className="editorial-app">
      <header className="masthead">
        <div className="masthead-title">
          <span className="masthead-overline mono">XP—GACHA / PRIVATE CATALOGUE</span>
          <h1>地下金库</h1>
          <p>墨白的个人馆藏目录与偏好索引</p>
        </div>
        <div className="issue-info mono">
          <span>ISSUE</span><b>VOL. 2026 / 08</b>
          <span>EDITION</span><b>FULL STACK C {backendVersion ? `· ${backendVersion}` : ""}</b>
          <span>INDEX</span><b>{catalogMetrics.items.toLocaleString()} ENTRIES</b>
          <span>STATUS</span><b>{backendStatus.toUpperCase()} · HISTORY {history.length}</b>
        </div>
      </header>

      <nav className="contents-nav" aria-label="主栏目">
        <span className="contents-label mono">CONTENTS</span>
        {navItems.map(([to, label], index) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/"}
            viewTransition
            className={({ isActive }) => isActive ? "contents-link contents-link-active" : "contents-link"}
          >
            <sup className="mono">{String(index + 1).padStart(2, "0")}</sup>{label}
          </NavLink>
        ))}
        <span className="nav-status mono">DB / {backendStatus === "online" ? "API" : backendStatus.toUpperCase()} · HISTORY / {history.length.toString().padStart(3, "0")}</span>
      </nav>

      <main id="main-content">
        <Suspense fallback={<div className="route-loading mono" role="status">LOADING SECTION…</div>}>
          <Routes>
            <Route path="/" element={<CatalogPage />} />
            <Route path="/detail/:id" element={<DetailPage />} />
            <Route path="/chat" element={<ChatPage />} />
            <Route path="/history" element={<HistoryPage />} />
            <Route path="/charts" element={<ChartsPage />} />
            <Route path="/admin" element={<AdminPage />} />
          </Routes>
        </Suspense>
      </main>
      {notice && <div className="editorial-toast" role="status"><span />{notice}</div>}
    </div>
  );
}
