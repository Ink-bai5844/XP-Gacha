import json
from collections import OrderedDict

INPUT_FILE = 'data_processing/111.json'
OUTPUT_FILE = 'data_processing/222.json'

def transform_semantic_map():
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误：未找到文件 {INPUT_FILE}")
        return
    except json.JSONDecodeError:
        print(f"错误：{INPUT_FILE} 不是有效的 JSON 文件")
        return

    result = {}
    seen = {}   # 记录每个映射值第一次出现的原文

    for original, mapped in data.items():
        # 如果这个映射值是第一次出现，记录原文
        if mapped not in seen:
            seen[mapped] = original
        
        first_original = seen[mapped]
        new_value = f"{mapped}({first_original})"
        
        result[original] = new_value

    # 保存新文件（保持键的原始顺序）
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    print(f"✅ 处理完成！")
    print(f"   原始文件：{INPUT_FILE}  →  新文件：{OUTPUT_FILE}")
    print(f"   共处理 {len(result)} 条映射")


if __name__ == "__main__":
    transform_semantic_map()