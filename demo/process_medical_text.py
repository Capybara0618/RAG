from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import requests

def read_medical_text(file_path):
    """读取医疗文本文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def split_text(text):
    """将文本分割成小块"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # 每个文本块的大小
        chunk_overlap=50,  # 文本块之间的重叠部分
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]  # 分割符
    )
    return text_splitter.split_text(text)

def get_embedding(text):
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
    
    # 分割文本
    chunks = split_text(text)
    print(f"文本被分割成 {len(chunks)} 个块")
    
    # 处理每个文本块
    results = []
    for i, chunk in enumerate(chunks, 1):
        print(f"正在处理第 {i}/{len(chunks)} 个文本块")
        embedding = get_embedding(chunk)
        results.append({
            "text": chunk,
            "embedding": embedding
        })
    
    # 保存结果
    with open('demo/medical_embeddings.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("处理完成，结果已保存到 medical_embeddings.json")

if __name__ == "__main__":
    process_and_save() 