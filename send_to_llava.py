import base64
import json
import os
import requests

def encode_image_base64(image_path): 
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image {image_path} not found.")
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def send_to_llava():
    context_path = "retrieved_context.json"
    if not os.path.exists(context_path):
        raise FileNotFoundError("retrieved_context.json not found.")

    with open(context_path, "r", encoding="utf-8") as f:
        context = json.load(f)

    query = context.get("query", "").strip()
    top_texts = context.get("top_texts", [])
    top_images = context.get("top_images", [])

    if not query:
        raise ValueError("Missing 'query' in JSON.")

    print(f"[INFO] Query: {query}")

    print("\n===== TEXT CHUNK RESPONSES =====\n")
    for i, (_score, text, page_num) in enumerate(top_texts[:3]):
        prompt = f"""Answer the following question based only on the text below:

Question: {query}

Text:
{text}
"""
        payload = {
            "model": "llava",
            "prompt": prompt,
            "stream": False
        }

        try:
            print(f"\n[INFO] Sending request for Text Chunk #{i + 1} (Page {page_num})...")
            response = requests.post("http://localhost:11434/api/generate", json=payload)
            response.raise_for_status()
            print(f"\n[Response - Text #{i + 1}]:\n{response.json().get('response', '[No response]')}")
        except Exception as e:
            print(f"[ERROR] Failed for text chunk #{i + 1}: {e}")

    print("\n===== IMAGE-BASED RESPONSE =====\n")
    image_b64_list = []
    for entry in top_images:
        page = entry.get("page")
        if page is None:
            continue
        image_path = f"page_{page}.png"
        try:
            image_b64_list.append(encode_image_base64(image_path))
        except FileNotFoundError as e:
            print(f"[WARNING] {e}")

    if not image_b64_list:
        print("[ERROR] No valid images found for visual response.")
        return

    image_prompt = f"""Answer the following question using only the visual content of the attached images.

Question: {query}
"""

    payload = {
        "model": "llava",
        "prompt": image_prompt,
        "images": image_b64_list,
        "stream": False
    }

    try:
        print(f"\n[INFO] Sending request for image-based reasoning...")
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        response.raise_for_status()
        print(f"\n[Response - Image]:\n{response.json().get('response', '[No response]')}")
    except Exception as e:
        print(f"[ERROR] Image reasoning failed: {e}")

if __name__ == "__main__":
    send_to_llava()