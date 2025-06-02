import json
import requests
from typing import List, Dict

def read_medical_text(file_path: str) -> str:
    """读取医疗文本文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def extract_diseases(text: str) -> List[Dict]:
    diseases = []
    current_name = None
    current_lines = []
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('## '):
            if current_name and current_lines:
                diseases.append({
                    'name': current_name,
                    'text': '\n'.join(current_lines).strip()
                })
            current_name = line[3:].strip()
            current_lines = []
        elif current_name:
            current_lines.append(line)
    # 最后一个疾病
    if current_name and current_lines:
        diseases.append({
            'name': current_name,
            'text': '\n'.join(current_lines).strip()
        })
    return diseases

def get_embedding(text: str) -> List[float]:
    """使用 Ollama 获取文本的向量表示"""
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": text}
    ).json()
    return response['embedding']

def process_and_save():
    """处理文本并保存结果"""
    # 读取文本
    text = read_medical_text('demo/medical_knowledge.txt')
    
    # 提取疾病信息
    diseases = extract_diseases(text)
    print(f"提取了 {len(diseases)} 个疾病")
    
    # 处理每个疾病
    results = []
    for disease in diseases:
        emb = get_embedding(disease['text'])
        results.append({
            'name': disease['name'],
            'text': disease['text'],
            'embedding': emb
        })
        print(f"已处理: {disease['name']}")
    
    # 保存结果
    with open('demo/medical_embeddings.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("处理完成，结果已保存到 medical_embeddings.json")

if __name__ == "__main__":
    process_and_save() 