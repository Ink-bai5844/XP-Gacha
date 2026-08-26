import { ArrowRight, Database, Download, ExternalLink, FileArchive, Upload } from "lucide-react";
import { Link } from "react-router-dom";

const DATASET_URL = "https://huggingface.co/datasets/Ink-bai/XP-Gacha-datasets";

export function CatalogImportGuide() {
  return (
    <section className="catalog-import-guide" aria-labelledby="catalog-import-guide-title">
      <header className="catalog-import-guide-heading">
        <div className="catalog-import-guide-mark" aria-hidden="true"><Database size={22} /></div>
        <div>
          <span className="mono">DATABASE / EMPTY</span>
          <h3 id="catalog-import-guide-title">先导入库存数据，再开始浏览</h3>
          <p>
            当前 MySQL 中没有馆藏条目。可以从 XP-Gacha 数据集页面下载
            <code>input_data.zip</code>，然后直接交给“附录”的一键导入，无需手动解压。
          </p>
        </div>
      </header>

      <div className="catalog-import-guide-actions" aria-label="数据导入快捷操作">
        <a href={DATASET_URL} target="_blank" rel="noreferrer" aria-label="打开 Hugging Face 数据集并下载 input_data.zip（新窗口）">
          <Download size={18} aria-hidden="true" />
          <span><strong>打开 Hugging Face 数据集</strong><small className="mono">下载 input_data.zip</small></span>
          <ExternalLink size={15} aria-hidden="true" />
        </a>
        <Link to="/admin?focus=import">
          <Upload size={18} aria-hidden="true" />
          <span><strong>前往一键导入</strong><small className="mono">附录 A / 导入词典与数据</small></span>
          <ArrowRight size={15} aria-hidden="true" />
        </Link>
      </div>

      <ol className="catalog-import-steps" aria-label="input_data.zip 导入步骤">
        <li>
          <span className="mono">01</span>
          <div><h4>打开数据集页面</h4><p>点击“打开 Hugging Face 数据集”，进入 <strong>Ink-bai / XP-Gacha-datasets</strong>；若页面要求登录或确认内容，请先按页面提示完成。</p></div>
        </li>
        <li>
          <span className="mono">02</span>
          <div><h4>下载 input_data.zip</h4><p>在数据集的文件列表中找到 <code>input_data.zip</code> 并下载到本机；下载完成后保持 ZIP 原样。</p></div>
        </li>
        <li>
          <span className="mono">03</span>
          <div><h4>选择 ZIP 并开始导入</h4><p>打开“前往一键导入”，在“导入包”中选择刚下载的 ZIP，数据库模式保持“增量写入 / 更新”，再点击“上传并一键导入”。</p></div>
        </li>
        <li>
          <span className="mono">04</span>
          <div><h4>等待完成并返回库存</h4><p>不要关闭启动窗口。页面显示“已导入 … 条”后，点击“返回库存目录”，即可开始检索和浏览。</p></div>
        </li>
      </ol>

      <aside className="catalog-import-note" role="note">
        <FileArchive size={18} aria-hidden="true" />
        <p><strong>不需要解压 ZIP。</strong>一键导入会自动查找其中的 CSV，并识别四个标准词典文件；以后重复导入时也建议使用“增量写入 / 更新”。</p>
      </aside>
    </section>
  );
}
