import csv
import json
import os
import glob

def load_semantic_map(json_file_path):
    """加载语义映射文件"""
    if not os.path.exists(json_file_path):
        print(f"警告：未找到映射文件 {json_file_path}，将保持原标签不变。")
        return {}
        
    with open(json_file_path, mode='r', encoding='utf-8') as f:
        return json.load(f)

def get_aggregated_tags(directory_path, semantic_map):
    """读取目录下所有CSV，仅保留词典中尚未收录的原始标签"""
    unique_tags = set()
    
    # 匹配文件夹下所有的 .csv 文件
    search_pattern = os.path.join(directory_path, '*.csv')
    csv_files = glob.glob(search_pattern)
    
    if not csv_files:
        print(f"提示：在 '{directory_path}' 目录下未找到任何 CSV 文件。")
        return unique_tags

    for csv_file_path in csv_files:
        # 打开csv文件
        with open(csv_file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            # 确保文件包含“标签”列头，避免 KeyError
            if not reader.fieldnames or '标签' not in reader.fieldnames:
                continue

            for row in reader:
                tags_string = row.get('标签', '')
                
                if tags_string:
                    # 按逗号分割字符串，并去除每个标签前后的空格，过滤掉空字符串
                    tags_list = [tag.strip() for tag in tags_string.split(',') if tag.strip()]
                    
                    for tag in tags_list:
                        # 仅导出语义词典中尚未覆盖到的原始标签
                        if tag not in semantic_map:
                            unique_tags.add(tag)
                
    return unique_tags

def export_tags_to_document(tags, output_file_path):
    """将聚合后的标签导出为文本文件"""
    output_dir = os.path.dirname(output_file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    sorted_tags = sorted(tags)
    with open(output_file_path, mode='w', encoding='utf-8-sig') as f:
        f.write(f"共找到 {len(sorted_tags)} 个不重复的标签：\n\n")
        for tag in sorted_tags:
            f.write(f"{tag}\n")

    return output_file_path

if __name__ == '__main__':
    # 文件夹路径和JSON映射文件路径
    target_directory = 'data/gallery_info'
    json_map_path = 'dictionaries/SEMANTIC_MAP.json'
    output_file_path = 'data_processing/aggregated_tags.txt'
    
    # 解析JSON映射表
    semantic_map = load_semantic_map(json_map_path)
    
    # 遍历所有CSV并获取聚合后的标签集合
    result_tags = get_aggregated_tags(target_directory, semantic_map)
    
    # 导出文档
    exported_file = export_tags_to_document(result_tags, output_file_path)

    # 打印结果
    sorted_tags = sorted(result_tags)
    print(f"共找到 {len(sorted_tags)} 个不重复的标签，已导出到：{exported_file}")
    print("以下为前 20 个标签预览：")
    for tag in sorted_tags[:20]:
        print(tag)
