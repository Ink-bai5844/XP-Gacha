import pandas as pd
import re
import html

def process_gallery_data(csv_file, txt_file, output_file):
    # 解析 all.txt 文件，提取链接和对应的文件名
    # 创建字典：{ '链接': '文件名' }
    url_to_name = {}
    
    # 正则表达式：
    # HREF="(.*?)" 匹配链接
    # >(.*?)</A> 匹配标签中间的文件名文字
    pattern = re.compile(r'HREF="(.*?)".*?>(.*?)</A>')

    try:
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    url = match.group(1)
                    # 使用 html.unescape 将 HTML 实体（如 &#39;）转回正常字符（如 '）
                    name = html.unescape(match.group(2))
                    url_to_name[url] = name
    except FileNotFoundError:
        print(f"错误：找不到文件 {txt_file}")
        return

    # 读取原始 CSV 表格
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"错误：找不到文件 {csv_file}")
        return

    # 匹配数据并插入新列
    # 根据“链接”列从字典中查找对应的“文件名”，如果找不到则填入空字符串
    file_names = df['链接'].map(url_to_name).fillna('')
    
    # 在第二列（索引为1）的位置插入“文件名”列
    df.insert(1, '文件名', file_names)

    # 保存处理后的结果
    # 使用 utf-8-sig 编码以确保 Excel 打开时不乱码
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"处理完成！已生成新表格：{output_file}")

if __name__ == "__main__":
    process_gallery_data('data/gallery_info_no_name/gallery_info_chinese.csv', 'data/local_data/all.txt', 'data/gallery_info/gallery_info_chinese_full.csv')