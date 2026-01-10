"""Rate limiter с retry логикой"""
import asyncio
import time
from collections import defaultdict
from typing import Optional, Callable, Any
from dataclasses import dataclass
from functools import wraps


@dataclass
class RateLimitConfig:
    """Конфигурация rate limiter"""
    max_requests_per_minute: int = 5
    max_requests_per_hour: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: int = 60


class RateLimiter:
    """Rate limiter с поддержкой retry логики"""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        
        # Хранение временных меток запросов
        # {user_id: [timestamp1, timestamp2, ...]}
        self.requests_per_user_minute: dict[int, list[float]] = defaultdict(list)
        self.requests_per_user_hour: dict[int, list[float]] = defaultdict(list)
        
        # Блокировка для потокобезопасности
        self.lock = asyncio.Lock()
    
    def _cleanup_old_requests(self, user_id: int):
        """Удалить старые записи из истории запросов"""
        now = time.time()
        
        # Очищаем минутные запросы (старше 60 секунд)
        self.requests_per_user_minute[user_id] = [
            ts for ts in self.requests_per_user_minute[user_id]
            if now - ts < 60
        ]
        
        # Очищаем часовые запросы (старше 3600 секунд)
        self.requests_per_user_hour[user_id] = [
            ts for ts in self.requests_per_user_hour[user_id]
            if now - ts < 3600
        ]
    
    def _can_make_request(self, user_id: int) -> tuple[bool, Optional[str]]:
        """Проверить, можно ли сделать запрос"""
        self._cleanup_old_requests(user_id)
        
        minute_count = len(self.requests_per_user_minute[user_id])
        hour_count = len(self.requests_per_user_hour[user_id])
        
        if minute_count >= self.config.max_requests_per_minute:
            return False, f"Превышен лимит запросов в минуту ({self.config.max_requests_per_minute})"
        
        if hour_count >= self.config.max_requests_per_hour:
            return False, f"Превышен лимит запросов в час ({self.config.max_requests_per_hour})"
        
        return True, None
    
    async def acquire(self, user_id: int) -> bool:
        """Попытка получить доступ к выполнению запроса"""
        async with self.lock:
            can_proceed, message = self._can_make_request(user_id)
            
            if can_proceed:
                now = time.time()
                self.requests_per_user_minute[user_id].append(now)
                self.requests_per_user_hour[user_id].append(now)
                return True
            
            return False
    
    async def execute_with_retry(
        self, 
        user_id: int, 
        func: Callable, 
        *args, 
        on_rate_limit: Optional[Callable] = None,
        **kwargs
    ) -> tuple[Any, bool]:
        """
        Выполнить функцию с retry логикой при rate limit
        
        Возвращает:
            tuple: (результат, success_flag)
        """
        for attempt in range(self.config.max_retries):
            # Попытка получить доступ
            if await self.acquire(user_id):
                try:
                    result = await func(*args, **kwargs)
                    return result, True
                except Exception as e:
                    # Ошибка выполнения - не считаем как rate limit
                    if attempt == self.config.max_retries - 1:
                        raise e
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
            
            # Rate limit сработал
            if on_rate_limit:
                await on_rate_limit(user_id, attempt)
            
            # Ожидание перед следующей попыткой
            wait_time = self.config.retry_delay * (2 ** attempt)  # Экспоненциальная задержка
            await asyncio.sleep(wait_time)
        
        # Все попытки исчерпаны
        return None, False


class AsyncRateLimiter:
    """Асинхронный rate limiter для глобального использования"""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.limiters: dict[int, RateLimiter] = {}
        self.global_limiter = RateLimiter(config)
    
    def get_limiter(self, user_id: int) -> RateLimiter:
        """Получить rate limiter для конкретного пользователя"""
        if user_id not in self.limiters:
            self.limiters[user_id] = RateLimiter(self.config)
        return self.limiters[user_id]
    
    async def execute(
        self,
        user_id: int,
        func: Callable,
        *args,
        on_rate_limit: Optional[Callable] = None,
        **kwargs
    ):
        """Выполнить функцию с rate limiting и retry"""
        limiter = self.get_limiter(user_id)
        result, success = await limiter.execute_with_retry(
            user_id, func, *args, on_rate_limit=on_rate_limit, **kwargs
        )
        
        if not success:
            raise RateLimitError(f"Исчерпаны все попытки для пользователя {user_id}")
        
        return result
    
    async def check_rate_limit(self, user_id: int) -> tuple[bool, Optional[str]]:
        """Проверить, можно ли сделать запрос (без выполнения функции)"""
        limiter = self.get_limiter(user_id)
        async with limiter.lock:
            return limiter._can_make_request(user_id)


class RateLimitError(Exception):
    """Исключение при превышении rate limit"""
    pass


def rate_limit_decorator(limiter: AsyncRateLimiter):
    """Декоратор для rate limiting"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Предполагаем, что первый аргумент - это update с user_id
            if len(args) > 0 and hasattr(args[0], 'message') and args[0].message and args[0].message.from_user:
                user_id = args[0].message.from_user.id
            elif len(args) > 0 and hasattr(args[0], 'from_user'):
                user_id = args[0].from_user.id
            else:
                raise ValueError("Не удалось определить user_id для rate limiting")
            
            return await limiter.execute(user_id, func, *args, **kwargs)
        
        return wrapper
    return decorator
