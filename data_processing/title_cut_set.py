import pandas as pd
import re
import json
from pathlib import Path
from collections import Counter
from janome.tokenizer import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_ROOT / 'data' / 'gallery_info' / 'gallery_info_gender_bender_full.csv'
OUTPUT_CSV = PROJECT_ROOT / 'data_processing' / 'title_words_frequency.csv'
TITLE_STOP_WORDS_PATH = PROJECT_ROOT / 'dictionaries' / 'TITLE_STOP_WORDS.txt'
TITLE_SEMANTIC_MAP_PATH = PROJECT_ROOT / 'dictionaries' / 'TITLE_SEMANTIC_MAP.json'

def load_title_stop_words(file_path):
    """从词典文件中提取被引号包裹的停用词。"""
    if not file_path.exists():
        print(f"警告: 找不到停用词词典 {file_path}，将使用空停用词集。")
        return set()

    content = file_path.read_text(encoding='utf-8')
    return set(re.findall(r"""['"]([^'"]+)['"]""", content))

def load_title_semantic_map(file_path):
    """加载标题语义映射词典。"""
    if not file_path.exists():
        print(f"警告: 找不到语义词典 {file_path}，将不做语义映射。")
        return {}

    with file_path.open(mode='r', encoding='utf-8') as file:
        return json.load(file)

def process_title_words():
    print("正在初始化分词器...")
    tokenizer = Tokenizer()
    title_stop_words = load_title_stop_words(TITLE_STOP_WORDS_PATH)
    title_semantic_map = load_title_semantic_map(TITLE_SEMANTIC_MAP_PATH)
    
    print(f"正在读取 {INPUT_CSV} ...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {INPUT_CSV}，请确保该脚本与 CSV 在同一目录下。")
        return
        
    # 处理空标题：如果标题为空，则使用文件名替代
    df['标题'] = df['标题'].fillna('')
    df['文件名'] = df['文件名'].fillna('')
    df['最终标题'] = df.apply(lambda row: row['文件名'] if row['标题'].strip() == '' else row['标题'], axis=1)
    
    all_extracted_words = []
    
    print("正在进行分词与特征提取，请稍候...")
    for title in df['最终标题']:
        if not isinstance(title, str) or not title.strip():
            continue
            
        # 移除方括号和圆括号内容
        clean_title = re.sub(r'\[.*?\]', ' ', title)
        clean_title = re.sub(r'\(.*?\)', ' ', clean_title)
        clean_title = clean_title.strip()
        
        # 分词与词性过滤
        for token in tokenizer.tokenize(clean_title):
            word = token.surface
            part_of_speech = token.part_of_speech.split(',')[0]
            
            # 仅保留名词、动词、形容词，过滤单字符(非中文字符时)和停用词
            if part_of_speech in ['名詞', '動詞', '形容詞'] and word not in title_stop_words and len(word) >= 1:
                # 语义映射
                mapped_word = title_semantic_map.get(word, word)
                all_extracted_words.append(mapped_word)
                
    # 统计词频
    word_freq = Counter(all_extracted_words)
    
    # 转换为 DataFrame 并导出
    print("\n正在生成统计结果...")
    result_df = pd.DataFrame(word_freq.items(), columns=['词汇', '频次'])
    result_df = result_df.sort_values(by='频次', ascending=False).reset_index(drop=True)
    
    result_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"完整词频统计已保存至: {OUTPUT_CSV}")
    
    # 在控制台打印 Top 50 供快速预览
    print("\n=== Top 50 高频词汇预览 ===")
    print(result_df.head(50).to_string(index=False))

if __name__ == "__main__":
    process_title_words()
