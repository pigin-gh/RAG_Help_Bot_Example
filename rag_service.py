"""
RAG сервис для работы с базой знаний.
Загружает базу знаний, разбивает на чанки и предоставляет поиск релевантных фрагментов.
"""
import sys
import os
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    # Fallback для обратной совместимости
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from tqdm import tqdm


class RAGService:
    """Сервис для работы с базой знаний через RAG"""
    
    def __init__(self, knowledge_base_path: str, vectorstore_dir: Optional[str] = None):
        """
        Инициализирует RAG сервис
        
        Args:
            knowledge_base_path: Путь к файлу с базой знаний
            vectorstore_dir: Директория для сохранения векторного индекса (по умолчанию: vectorstore/)
        """
        self.knowledge_base_path = Path(knowledge_base_path)
        self.vectorstore_dir = Path(vectorstore_dir) if vectorstore_dir else Path(__file__).parent / "vectorstore"
        self.vectorstore_dir.mkdir(exist_ok=True)
        self.vectorstore = None
        self.embeddings = None
        self._initialize()
    
    def _get_index_path(self) -> Path:
        """Возвращает путь к сохраненному индексу"""
        # Используем имя файла базы знаний для создания уникального пути
        kb_name = self.knowledge_base_path.stem
        return self.vectorstore_dir / f"{kb_name}_faiss_index"
    
    def _should_rebuild_index(self) -> bool:
        """
        Проверяет, нужно ли пересоздавать индекс
        
        Returns:
            True если индекс нужно пересоздать, False если можно загрузить существующий
        """
        index_path = self._get_index_path()
        
        # Если индекс не существует, нужно создать
        if not index_path.exists():
            return True
        
        # Проверяем, изменился ли файл базы знаний
        kb_mtime = os.path.getmtime(self.knowledge_base_path)
        
        # Проверяем время модификации индекса
        # Ищем файл index.faiss (основной файл индекса)
        index_file = index_path / "index.faiss"
        if not index_file.exists():
            return True
        
        index_mtime = os.path.getmtime(index_file)
        
        # Если база знаний новее индекса, нужно пересоздать
        return kb_mtime > index_mtime
    
    def _initialize(self):
        """Загружает базу знаний и создает/загружает векторное хранилище"""
        print("\n" + "="*60)
        print("📚 ИНИЦИАЛИЗАЦИЯ RAG СЕРВИСА")
        print("="*60)
        
        # Инициализация embeddings (нужна всегда)
        print("\n[1/4] Инициализация модели embeddings...")
        print("   (Это может занять 1-3 минуты при первом запуске)")
        print("   Модель: intfloat/multilingual-e5-base")
        print("   Загрузка модели из HuggingFace...")
        sys.stdout.flush()
        
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="intfloat/multilingual-e5-base",
                model_kwargs={'device': 'cpu'}
            )
            print("   ✓ Модель успешно загружена!")
        except Exception as e:
            print(f"   ✗ Ошибка при загрузке модели: {e}")
            raise
        
        index_path = self._get_index_path()
        
        # Проверяем, нужно ли пересоздавать индекс
        if not self._should_rebuild_index():
            # Загружаем существующий индекс
            print(f"\n[2/4] Загрузка сохраненного векторного индекса...")
            print(f"   Путь: {index_path}")
            try:
                self.vectorstore = FAISS.load_local(
                    str(index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print("   ✓ Индекс загружен успешно!")
                print("\n" + "="*60)
                print("✅ БАЗА ЗНАНИЙ ГОТОВА К ИСПОЛЬЗОВАНИЮ!")
                print("="*60 + "\n")
                return
            except Exception as e:
                print(f"   ⚠ Ошибка при загрузке индекса: {e}")
                print("   Будет создан новый индекс...")
        
        # Создаем новый индекс
        print("\n[2/4] Загрузка базы знаний из файла...")
        loader = TextLoader(
            str(self.knowledge_base_path),
            encoding='utf-8'
        )
        documents = loader.load()
        print(f"✓ Файл загружен: {len(documents)} документ(ов)")
        
        # Разбиение на чанки
        print("\n[3/4] Разбиение текста на фрагменты...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"✓ База знаний разбита на {len(chunks)} фрагментов")
        
        # Создание векторного хранилища
        print(f"\n[4/4] Создание векторного индекса для {len(chunks)} фрагментов...")
        print("   (Это может занять 1-3 минуты, пожалуйста, подождите...)")
        print("   Создание векторных представлений и индекса FAISS...")
        sys.stdout.flush()
        
        try:
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
            print("   ✓ Индекс создан успешно!")
            
            # Сохраняем индекс на диск
            print(f"\n[5/5] Сохранение индекса на диск...")
            print(f"   Путь: {index_path}")
            self.vectorstore.save_local(str(index_path))
            print("   ✓ Индекс сохранен!")
        except Exception as e:
            print(f"\n   ✗ Ошибка при создании индекса: {e}")
            raise
        
        print("\n" + "="*60)
        print("✅ БАЗА ЗНАНИЙ ГОТОВА К ИСПОЛЬЗОВАНИЮ!")
        print("="*60 + "\n")
    
    def get_relevant_context(self, query: str, k: int = 3) -> List[Document]:
        """
        Получает релевантные фрагменты из базы знаний
        
        Args:
            query: Поисковый запрос
            k: Количество возвращаемых фрагментов
            
        Returns:
            Список релевантных документов
        """
        if self.vectorstore is None:
            raise ValueError("Векторное хранилище не инициализировано")
        
        # Поиск релевантных документов
        docs = self.vectorstore.similarity_search(query, k=k)
        return docs
    
    def get_relevant_context_as_text(self, query: str, k: int = 3) -> str:
        """
        Получает релевантные фрагменты в виде текста
        
        Args:
            query: Поисковый запрос
            k: Количество возвращаемых фрагментов
            
        Returns:
            Текст с релевантными фрагментами
        """
        docs = self.get_relevant_context(query, k)
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"[Фрагмент {i}]\n{doc.page_content}\n")

        # Выводим выбранные фрагменты в консоль, чтобы видеть, что отправляем в LLM
        print("\n--- RAG: выбранные фрагменты ---")
        for i, part in enumerate(context_parts, 1):
            print(f"[{i}] {part}")
        print("--- Конец фрагментов ---\n")

        return "\n".join(context_parts)

