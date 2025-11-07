# utils/cache.py
import os
import requests
from pathlib import Path
import hashlib

CACHE_DIR = Path("cache/pdfs")
CACHE_DIR.mkdir(exist_ok=True)

def download_pdf_cached(pdf_url: str, timeout: int = 15) -> Path:
    """
    Скачивает PDF по URL и кэширует его.
    Возвращает путь к файлу.
    """
    # Генерируем имя файла по хешу URL
    filename = hashlib.md5(pdf_url.encode()).hexdigest() + ".pdf"
    filepath = CACHE_DIR / filename

    if filepath.exists():
        print(f"📄 Используем кэш: {filepath}")
        return filepath

    print(f"📥 Скачиваем PDF: {pdf_url}")
    response = requests.get(pdf_url, timeout=timeout)
    response.raise_for_status()

    with open(filepath, "wb") as f:
        f.write(response.content)

    print(f"💾 Сохранён в кэш: {filepath}")
    return filepath