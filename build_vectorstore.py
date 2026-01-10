"""
Утилита для разовой векторизации базы знаний и сохранения индекса FAISS.

Запуск:
    uv run build_vectorstore.py
или:
    python build_vectorstore.py

После успешного запуска в каталоге `vectorstore/` появится директория
`baza-znanii-ecvi_faiss_index/`, которую можно скопировать на сервер и
использовать вместе с Telegram-ботом без повторной векторизации.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from rag_service import RAGService


def main() -> None:
    # Загружаем переменные окружения (на данный момент для RAG они не критичны,
    # но держим единообразие с остальными скриптами проекта)
    load_dotenv(find_dotenv())

    knowledge_base_path = Path(__file__).parent / "baza-znanii-ecvi.txt"
    print("=== Построение векторного индекса для базы знаний ===")
    print(f"Файл базы знаний: {knowledge_base_path}")

    # Создание RAGService автоматически создаст/обновит FAISS-индекс
    # Используем те же параметры, что и в основном боте
    _ = RAGService(
        str(knowledge_base_path),
        chunk_size=1000,
        chunk_overlap=200,
        use_hybrid_search=True  # Создаст BM25 индекс для гибридного поиска
    )

    print("\nГотово. Индекс сохранён в каталоге `vectorstore/`.")
    print("Скопируйте папку `vectorstore/` на сервер рядом с кодом проекта.")


if __name__ == "__main__":
    main()



