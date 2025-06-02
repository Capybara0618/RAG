import json
import numpy as np
import requests
from typing import List, Dict
import sys

def load_embeddings(file_path: str) -> List[Dict]:
    print(f"[流程] 正在加载向量化数据: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"[流程] 成功加载 {len(data)} 条疾病数据")
        return data
    except Exception as e:
        print(f"[错误] 加载向量化数据失败: {e}")
        sys.exit(1)

def get_embedding(text: str) -> List[float]:
    print(f"[流程] 正在获取 embedding，文本前30字: {text[:30]}...")
    try:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text},
            timeout=10
        )
        response.raise_for_status()
        emb = response.json()['embedding']
        print(f"[流程] embedding 获取成功，长度: {len(emb)}")
        return emb
    except Exception as e:
        print(f"[错误] 获取 embedding 失败: {e}")
        sys.exit(1)

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    sim = float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    return sim

def retrieve_context(query: str, data: List[Dict], top_k: int = 3) -> List[Dict]:
    print(f"[流程] 正在检索相关疾病资料，问题: {query}")
    query_emb = get_embedding(query)
    scored = []
    for disease in data:
        sim = cosine_similarity(query_emb, disease['embedding'])
        scored.append((sim, disease))
    scored.sort(key=lambda x: x[0], reverse=True)
    top_contexts = [d for _, d in scored[:top_k]]
    print(f"[流程] 检索到 top{top_k} 疾病: {[c['name'] for c in top_contexts]}")
    return top_contexts

def build_prompt(question: str, contexts: List[Dict]) -> str:
    print(f"[流程] 正在构建 prompt，包含 {len(contexts)} 条 context")
    context_text = '\n\n'.join([f"疾病：{c['name']}\n内容：{c['text']}" for c in contexts])
    prompt = f"你是专业的医疗助手。请根据以下疾病资料，结合用户问题，给出详细、准确、通俗的医学解答。\n\n【疾病资料】\n{context_text}\n\n【用户问题】\n{question}\n\n【医学解答】"
    print(f"[流程] prompt 构建完成，长度: {len(prompt)}")
    return prompt

def ask_deepseek(prompt: str) -> str:
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "deepseek-r1:1.5b",
        "prompt": prompt,
        "stream": False
    }
    print(f"[流程] 正在请求 deepseek，大模型 payload: {payload['model']}")
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        print(f"[流程] deepseek 返回: {str(result)[:100]}...")
        return result.get('response', result.get('text', str(result)))
    except Exception as e:
        print(f"[错误] DeepSeek 生成失败: {e}")
        return f"[DeepSeek 生成失败]: {e}"

def main():
    print("[流程] 正在加载医疗知识库...")
    data = load_embeddings('demo/medical_embeddings.json')
    print(f"[流程] 已加载 {len(data)} 个疾病条目。\n")
    print("欢迎使用 DeepSeek 医疗问答系统！输入 quit 退出。\n")
    while True:
        try:
            question = input("请输入您的医疗问题：").strip()
            if not question:
                continue
            if question.lower() == 'quit':
                print("感谢使用，再见！")
                break
            print("[流程] 用户输入问题，开始检索和生成...")
            contexts = retrieve_context(question, data, top_k=3)
            prompt = build_prompt(question, contexts)
            answer = ask_deepseek(prompt)
            print("\n【医学解答】\n" + answer + "\n")
        except KeyboardInterrupt:
            print("\n程序中断，退出。")
            break
        except Exception as e:
            print(f"[错误] 发生错误: {e}")
            continue

if __name__ == "__main__":
    main() 