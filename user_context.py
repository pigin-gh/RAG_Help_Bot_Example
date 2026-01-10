"""Управление контекстом пользователей бота"""
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

CONTEXT_FILE = Path(__file__).parent / "user_contexts.json"


class UserContext:
    """Хранение данных пользователей"""
    
    def __init__(self):
        self.contexts: Dict[int, Dict[str, Any]] = {}
        self._load_contexts()
    
    def _load_contexts(self):
        """Загружает контексты из файла"""
        if CONTEXT_FILE.exists():
            try:
                with open(CONTEXT_FILE, 'r', encoding='utf-8') as f:
                    self.contexts = json.load(f)
                    self.contexts = {int(k): v for k, v in self.contexts.items()}
            except Exception:
                self.contexts = {}
    
    def _save_contexts(self):
        """Сохраняет контексты в файл"""
        try:
            with open(CONTEXT_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.contexts, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    
    def get_context(self, user_id: int) -> Dict[str, Any]:
        """Получает контекст пользователя"""
        return self.contexts.get(user_id, {
            'state': 'active',
            'created_at': None,
            'updated_at': None,
            'last_questions': []
        })
    
    def update_context(self, user_id: int, **kwargs):
        """Обновляет контекст пользователя"""
        if user_id not in self.contexts:
            self.contexts[user_id] = {
                'state': 'active',
                'created_at': datetime.now().isoformat(),
                'updated_at': None,
                'last_questions': []
            }
        
        for key, value in kwargs.items():
            self.contexts[user_id][key] = value
        
        self.contexts[user_id]['updated_at'] = datetime.now().isoformat()
        self._save_contexts()
    
    def clear_context(self, user_id: int):
        """Очищает контекст пользователя"""
        if user_id in self.contexts:
            del self.contexts[user_id]
            self._save_contexts()
    
    def add_question(self, user_id: int, question: str):
        """Добавляет вопрос в историю"""
        if user_id not in self.contexts:
            self.update_context(user_id)
        
        if 'last_questions' not in self.contexts[user_id]:
            self.contexts[user_id]['last_questions'] = []
        
        self.contexts[user_id]['last_questions'].append({
            'question': question,
            'timestamp': datetime.now().isoformat()
        })
        
        # Ограничиваем последними 10 вопросами
        self.contexts[user_id]['last_questions'] = self.contexts[user_id]['last_questions'][-10:]
        self._save_contexts()


user_context_manager = UserContext()
