import { useState } from "react";
import { PreferenceCharts } from "../components/PreferenceCharts";

export function ChartsPage() {
  const [scope, setScope] = useState<"global" | "history">("global");
  return (
    <div className="charts-page">
      <header className="page-intro page-intro-split">
        <div><span className="section-code">SECTION 05 / DATA JOURNAL</span><h2>偏好图表</h2></div>
        <p>查看全局与历史偏好的标签、作者和标题词排行，支持 Top 15 图表与 Top 150 明细。</p>
      </header>
      <div className="scope-tabs" role="tablist" aria-label="图表数据范围">
        <button type="button" role="tab" aria-selected={scope === "global"} onClick={() => setScope("global")}>全局偏好数据</button>
        <button type="button" role="tab" aria-selected={scope === "history"} onClick={() => setScope("history")}>用户历史偏好</button>
      </div>
      <PreferenceCharts scope={scope} />
    </div>
  );
}
