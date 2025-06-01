import requests

texts = ["这是第一个测试文本", "这是第二个不同的文本"]

for text in texts:
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": text}
    ).json()
    embedding = response['embedding']
    print(f"\n文本: {text}")
    print(f"向量维度: {len(embedding)}")
    print(f"向量前5个值: {embedding[:5]}")
