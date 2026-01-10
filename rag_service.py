"""
RAG сервис для работы с базой знаний.
Загружает базу знаний, разбивает на чанки и предоставляет поиск релевантных фрагментов.
"""
import sys
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

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
    
    def __init__(self, knowledge_base_path: str, vectorstore_dir: Optional[str] = None,
                 chunk_size: int = 1000, chunk_overlap: int = 200,
                 use_hybrid_search: bool = True):
        """
        Инициализирует RAG сервис
        
        Args:
            knowledge_base_path: Путь к файлу с базой знаний
            vectorstore_dir: Директория для сохранения векторного индекса (по умолчанию: vectorstore/)
            chunk_size: Размер чанков для разбиения текста
            chunk_overlap: Перекрытие между чанками
            use_hybrid_search: Использовать гибридный поиск (векторный + BM25)
        """
        self.knowledge_base_path = Path(knowledge_base_path)
        self.vectorstore_dir = Path(vectorstore_dir) if vectorstore_dir else Path(__file__).parent / "vectorstore"
        self.vectorstore_dir.mkdir(exist_ok=True)
        
        # Настройки разбиения
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_hybrid_search = use_hybrid_search
        
        # Компоненты
        self.vectorstore = None
        self.embeddings = None
        self.bm25_index = None
        self.chunks_for_bm25 = None
        self.documents = None  # Сохраняем документы для BM25
        
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
    
    def _build_bm25_index(self):
        """Строит BM25 индекс для гибридного поиска"""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            print("   ⚠ rank-bm25 не установлен. Установите: pip install rank-bm25")
            print("   Будет использован только векторный поиск.")
            self.bm25_index = None
            return

        # Подготавливаем документы для BM25
        self.chunks_for_bm25 = [doc.page_content.split() for doc in self.documents]
        self.bm25_index = BM25Okapi(self.chunks_for_bm25)
    
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
                
                # Для гибридного поиска нужно пересоздать BM25 индекс
                # Но документы не загружаются из сохраненного индекса, поэтому
                # гибридный поиск будет недоступен после загрузки
                if self.use_hybrid_search:
                    print("   ⚠ Гибридный поиск недоступен при загрузке сохраненного индекса.")
                    print("   Для использования гибридного поиска пересоздайте индекс.")
                
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
        print(f"   ✓ Файл загружен: {len(documents)} документ(ов)")
        
        # Разбиение на чанки
        print("\n[3/4] Разбиение текста на фрагменты...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],
            keep_separator=True,
        )
        chunks = text_splitter.split_documents(documents)
        
        # Добавляем метаданные к каждому чанку
        for chunk in chunks:
            if 'source' not in chunk.metadata:
                chunk.metadata['source'] = str(self.knowledge_base_path.name)
            chunk.metadata['document_type'] = 'knowledge_base'
        
        self.documents = chunks
        print(f"   ✓ База знаний разбита на {len(chunks)} фрагментов")
        
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
            
            # Строим BM25 индекс для гибридного поиска
            if self.use_hybrid_search:
                print("\n[6/6] Создание BM25 индекса для гибридного поиска...")
                self._build_bm25_index()
                if self.bm25_index:
                    print("   ✓ BM25 индекс готов")
                else:
                    print("   ⚠ BM25 индекс не создан (библиотека не установлена)")
        except Exception as e:
            print(f"\n   ✗ Ошибка при создании индекса: {e}")
            raise
        
        print("\n" + "="*60)
        print("✅ БАЗА ЗНАНИЙ ГОТОВА К ИСПОЛЬЗОВАНИЮ!")
        print("="*60 + "\n")
    
    def _advanced_rerank_results(self, query: str, results: List[Tuple[Document, float]],
                               top_k: int = None, user_context: dict = None) -> List[Tuple[Document, float]]:
        """Улучшенный re-ranking с множеством факторов"""
        if top_k is None:
            top_k = len(results)

        query_lower = query.lower()
        query_words = set(re.findall(r'\b\w+\b', query_lower))

        reranked = []
        for doc, base_score in results:
            factors = self._calculate_advanced_rerank_factors(query, query_words, doc, user_context)

            # Взвешенная комбинация факторов
            final_score = (
                base_score * 0.5 +           # Базовый векторный/BM25 score
                factors['word_overlap'] * 0.2 +     # Совпадение слов
                factors['semantic_match'] * 0.15 +  # Семантическое совпадение
                factors['authority'] * 0.1 +         # Авторитет документа
                factors['recency'] * 0.05             # Актуальность
            )

            reranked.append((doc, final_score))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]

    def _calculate_advanced_rerank_factors(self, query: str, query_words: set,
                                         doc: Document, user_context: dict = None) -> dict:
        """Рассчитывает различные факторы для улучшенного re-ranking"""
        content = doc.page_content.lower()
        metadata = doc.metadata

        factors = {}

        # 1. Word overlap (совпадение слов)
        content_words = set(re.findall(r'\b\w+\b', content))
        factors['word_overlap'] = len(query_words & content_words) / max(len(query_words), 1)

        # 2. Важные термины (длинные слова, специфические термины)
        important_words = [w for w in query_words if len(w) > 4]
        important_overlap = sum(1 for term in important_words if term in content)
        factors['semantic_match'] = important_overlap / max(len(important_words), 1) if important_words else 0.5

        # 3. Авторитет документа (базовое значение для простой базы знаний)
        factors['authority'] = 0.7  # Средний вес для базы знаний

        # 4. Актуальность (базовое значение)
        factors['recency'] = 0.5

        # 5. Контекст пользователя (если применимо)
        if user_context:
            # Можно добавить логику персонализации
            pass

        return factors

    def _hybrid_search(self, query: str, k: int = 5, vector_weight: float = 0.7) -> List[Tuple[Document, float]]:
        """Гибридный поиск: векторный + BM25"""
        if self.vectorstore is None:
            raise ValueError("Векторное хранилище не инициализировано")
        
        # 1. Векторный поиск
        vector_results = self.vectorstore.similarity_search_with_score(query, k=k * 2)
        
        # 2. BM25 поиск (если доступен)
        if self.use_hybrid_search and self.bm25_index is not None and self.documents is not None:
            try:
                query_tokens = query.lower().split()
                bm25_scores = self.bm25_index.get_scores(query_tokens)
                
                # Создаем словарь для быстрого доступа к BM25 scores по тексту документа
                bm25_scores_dict = {}
                for idx, doc_score in enumerate(bm25_scores):
                    if idx < len(self.documents):
                        doc = self.documents[idx]
                        doc_key = (doc.page_content[:100] + 
                                  doc.metadata.get('source', '')[:50])
                        bm25_scores_dict[doc_key] = doc_score
                
                # Комбинируем результаты
                combined = []
                for doc, distance in vector_results:
                    vector_score = 1 / (1 + distance)  # Конвертируем расстояние в оценку
                    
                    # Находим соответствующий BM25 score
                    doc_key = (doc.page_content[:100] + 
                              doc.metadata.get('source', '')[:50])
                    bm25_score = bm25_scores_dict.get(doc_key, 0.0)
                    
                    # Если не нашли точное совпадение, ищем по началу текста
                    if bm25_score == 0.0:
                        for key, score in bm25_scores_dict.items():
                            if doc.page_content[:50] in key or key[:50] in doc.page_content[:100]:
                                bm25_score = score
                                break
                    
                    # Нормализуем BM25 score
                    if len(bm25_scores) > 0:
                        max_bm25 = float(max(bm25_scores))
                    else:
                        max_bm25 = 1.0
                    normalized_bm25 = (bm25_score / max_bm25) if max_bm25 > 0 else 0.0
                    
                    # Reciprocal Rank Fusion (RRF) - комбинация рангов
                    rrf_score = (vector_weight * vector_score) + ((1 - vector_weight) * normalized_bm25)
                    combined.append((doc, rrf_score))
                
                combined.sort(key=lambda x: x[1], reverse=True)
                return combined
            except Exception as e:
                print(f"   ⚠ Ошибка при гибридном поиске, используем только векторный: {e}")
                import traceback
                traceback.print_exc()
        
        # Fallback: только векторный поиск
        results = []
        for doc, distance in vector_results:
            score = 1 / (1 + distance)
            results.append((doc, score))
        
        return results
    
    def get_relevant_context_with_scores(self, query: str, k: int = 3, score_threshold: float = 0.5, 
                                        user_context: dict = None):
        """Получает релевантные фрагменты с оценками релевантности"""
        if self.vectorstore is None:
            raise ValueError("Векторное хранилище не инициализировано")
        
        # Используем гибридный поиск или обычный векторный
        if self.use_hybrid_search and self.bm25_index is not None and self.documents is not None:
            results = self._hybrid_search(query, k=k * 2, vector_weight=0.7)
        else:
            # Обычный векторный поиск
            vector_results = self.vectorstore.similarity_search_with_score(query, k=k * 2)
            results = []
            for doc, distance in vector_results:
                score = 1 / (1 + distance)
                results.append((doc, score))
        
        # Фильтруем по порогу
        filtered_results = [(doc, score) for doc, score in results if score >= score_threshold]
        
        # Применяем улучшенный re-ranking
        if len(filtered_results) > k:
            filtered_results = self._advanced_rerank_results(query, filtered_results, top_k=k, user_context=user_context)
        else:
            # Если результатов меньше k, все равно применяем re-ranking для улучшения порядка
            filtered_results = self._advanced_rerank_results(query, filtered_results, user_context=user_context)
        
        return filtered_results
    
    def get_relevant_context(self, query: str, k: int = 3) -> List[Document]:
        """
        Получает релевантные фрагменты из базы знаний
        
        Args:
            query: Поисковый запрос
            k: Количество возвращаемых фрагментов
            
        Returns:
            Список релевантных документов
        """
        results = self.get_relevant_context_with_scores(query, k=k)
        return [doc for doc, score in results]
    
    def get_relevant_context_as_text(self, query: str, k: int = 3, score_threshold: float = 0.5, 
                                     user_context: dict = None) -> tuple[str, bool]:
        """
        Получает релевантные фрагменты в виде структурированного текста
        
        Args:
            query: Поисковый запрос
            k: Количество возвращаемых фрагментов
            score_threshold: Минимальный порог релевантности
            user_context: Контекст пользователя для персонализации
            
        Returns:
            Кортеж (текст контекста, флаг успешности)
        """
        results = self.get_relevant_context_with_scores(query, k, score_threshold, user_context)
        
        if not results:
            return "", False
        
        context_parts = [
            "Ниже приведены выдержки из базы знаний. Используй их для ответа пользователю.\n"
        ]
        
        # Группируем по документам для лучшей структуризации
        seen_docs = {}
        
        for i, (doc, score) in enumerate(results, 1):
            # Получаем название документа из метаданных
            doc_name = doc.metadata.get('source', 'база знаний')
            
            # Очищаем название от расширений файлов
            if '.' in doc_name:
                doc_name = doc_name.rsplit('.', 1)[0]
            
            if score >= 0.8:
                relevance_label = "высокая"
            elif score >= 0.6:
                relevance_label = "средняя"
            else:
                relevance_label = "низкая"
            
            # Формируем структурированный вывод с явным указанием источника
            context_parts.append(
                f"---\n"
                f"Фрагмент {i}\n"
                f"Релевантность: {relevance_label} ({score:.1%})\n"
                f"Текст:\n{doc.page_content}\n"
            )
            
            # Собираем уникальные источники
            if doc_name not in seen_docs:
                seen_docs[doc_name] = score

        return "\n".join(context_parts), True
