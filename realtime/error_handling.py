"""
Comprehensive error handling and fallback mechanisms for the audio analysis system.

This module provides robust error handling strategies including:
- Circuit breaker pattern for repeated failures
- Retry mechanisms with exponential backoff
- Graceful degradation to minimal functionality
- Error logging and alerting
- Cross-component error propagation
- Resource pressure management
"""

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

# Configure logging
logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Enumeration of error types."""

    INITIALIZATION = "initialization"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    COMMUNICATION_FAILURE = "communication_failure"
    VALIDATION_ERROR = "validation_error"
    TIMEOUT = "timeout"
    CIRCUIT_BREAKER = "circuit_breaker"
    FALLBACK_ACTIVATED = "fallback_activated"


@dataclass
class ErrorContext:
    """Context information for error handling."""

    error_type: ErrorType
    component_name: str
    operation: str
    timestamp: float = field(default_factory=time.time)
    error_details: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    fallback_attempted: bool = False
    circuit_breaker_open: bool = False


class CircuitBreaker:
    """Circuit breaker pattern for handling repeated failures."""

    def __init__(
        self, failure_threshold: int = 5, recovery_timeout: float = 30.0, timeout_window: int = 60
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit breaker
            recovery_timeout: Time to wait before attempting recovery
            timeout_window: Time window for counting failures
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.timeout_window = timeout_window

        self.failure_count = 0
        self.last_failure_time = 0
        self.failure_times: List[float] = []
        self.state = "closed"  # closed, open, half-open
        self.lock = threading.RLock()

    def record_failure(self):
        """Record a failure and update circuit breaker state."""
        with self.lock:
            current_time = time.time()
            self.failure_times.append(current_time)

            # Clean up old failures
            self.failure_times = [
                t for t in self.failure_times if current_time - t <= self.timeout_window
            ]

            self.failure_count = len(self.failure_times)

            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                self.last_failure_time = current_time
                logger.warning(
                    f"Circuit breaker opened for component after {self.failure_count} failures"
                )

    def record_success(self):
        """Record a success and reset failure count."""
        with self.lock:
            self.failure_count = 0
            self.failure_times = []
            if self.state == "half-open":
                self.state = "closed"
            logger.info("Circuit breaker reset after successful operation")

    def allow_request(self) -> bool:
        """Check if request is allowed based on circuit breaker state."""
        with self.lock:
            if self.state == "closed":
                return True

            if self.state == "open":
                # Check if recovery timeout has passed
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    logger.info("Circuit breaker transitioning to half-open state")
                    self.state = "half-open"
                    return True
                return False

            # Half-open state - allow one request
            return True


class RetryPolicy:
    """Retry policy with exponential backoff."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 0.1,
        max_delay: float = 10.0,
        backoff_factor: float = 2.0,
    ):
        """
        Initialize retry policy.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between retries
            max_delay: Maximum delay between retries
            backoff_factor: Exponential backoff factor
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        delay = self.base_delay * (self.backoff_factor**attempt)
        return min(delay, self.max_delay)


class ErrorContextManager:
    """Manages error contexts across components."""

    def __init__(self):
        self.contexts: Dict[str, ErrorContext] = {}
        self.lock = threading.RLock()
        self.error_callbacks: Dict[str, List[Callable]] = {}

    def create_context(
        self, component_name: str, operation: str, error_type: ErrorType, **kwargs
    ) -> ErrorContext:
        """Create a new error context."""
        context = ErrorContext(
            error_type=error_type,
            component_name=component_name,
            operation=operation,
            error_details=kwargs,
        )

        with self.lock:
            context_key = f"{component_name}:{operation}"
            self.contexts[context_key] = context

        return context

    def update_context(self, component_name: str, operation: str, **updates):
        """Update existing error context."""
        context_key = f"{component_name}:{operation}"

        with self.lock:
            if context_key in self.contexts:
                context = self.contexts[context_key]
                for key, value in updates.items():
                    if hasattr(context, key):
                        setattr(context, key, value)

    def get_context(self, component_name: str, operation: str) -> Optional[ErrorContext]:
        """Get error context for given component and operation."""
        context_key = f"{component_name}:{operation}"

        with self.lock:
            return self.contexts.get(context_key)

    def register_error_callback(self, component_name: str, callback: Callable):
        """Register error callback for component."""
        with self.lock:
            if component_name not in self.error_callbacks:
                self.error_callbacks[component_name] = []
            self.error_callbacks[component_name].append(callback)

    def trigger_error_callbacks(self, component_name: str, context: ErrorContext):
        """Trigger error callbacks for component."""
        with self.lock:
            if component_name in self.error_callbacks:
                for callback in self.error_callbacks[component_name]:
                    try:
                        callback(context)
                    except Exception as e:
                        logger.error(f"Error in error callback: {e}")


class FallbackRegistry:
    """Registry of fallback operations for various components."""

    def __init__(self):
        self.fallbacks: Dict[str, Dict[str, Callable]] = {}
        self.lock = threading.RLock()

    def register_fallback(self, component_name: str, operation: str, fallback_func: Callable):
        """Register a fallback function for component operation."""
        with self.lock:
            if component_name not in self.fallbacks:
                self.fallbacks[component_name] = {}
            self.fallbacks[component_name][operation] = fallback_func

    def get_fallback(self, component_name: str, operation: str) -> Optional[Callable]:
        """Get fallback function for component operation."""
        with self.lock:
            if component_name in self.fallbacks and operation in self.fallbacks[component_name]:
                return self.fallbacks[component_name][operation]
            return None

    def has_fallback(self, component_name: str, operation: str) -> bool:
        """Check if fallback exists for component operation."""
        return self.get_fallback(component_name, operation) is not None


class ResourceMonitor:
    """Monitors system resources and provides warnings."""

    def __init__(self, warning_thresholds: Dict[str, float] = None):
        """Initialize resource monitor."""
        self.warning_thresholds = warning_thresholds or {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "gpu_memory": 90.0,
            "disk_space": 90.0,
        }
        self.resource_history: Dict[str, List[float]] = {}
        self.lock = threading.RLock()

    def check_resource_thresholds(self, current_metrics: Dict[str, float]) -> List[str]:
        """Check which resources are above warning thresholds."""
        warnings = []

        with self.lock:
            for resource, value in current_metrics.items():
                if resource in self.warning_thresholds:
                    if value >= self.warning_thresholds[resource]:
                        warnings.append(resource)

                    # Update history
                    if resource not in self.resource_history:
                        self.resource_history[resource] = []
                    self.resource_history[resource].append(value)

                    # Keep only last 100 measurements
                    if len(self.resource_history[resource]) > 100:
                        self.resource_history[resource] = self.resource_history[resource][-100:]

        return warnings

    def get_resource_trends(self) -> Dict[str, str]:
        """Get resource usage trends."""
        trends = {}

        with self.lock:
            for resource, history in self.resource_history.items():
                if len(history) >= 2:
                    if history[-1] > history[0]:
                        trends[resource] = "increasing"
                    elif history[-1] < history[0]:
                        trends[resource] = "decreasing"
                    else:
                        trends[resource] = "stable"

        return trends


class ErrorHandler:
    """Main error handler class."""

    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_policies: Dict[str, RetryPolicy] = {}
        self.context_manager = ErrorContextManager()
        self.fallback_registry = FallbackRegistry()
        self.resource_monitor = ResourceMonitor()
        self.global_error_handlers: List[Callable] = []
        self.lock = threading.RLock()

    def register_circuit_breaker(self, component_name: str, **kwargs) -> CircuitBreaker:
        """Register a circuit breaker for component."""
        with self.lock:
            if component_name not in self.circuit_breakers:
                self.circuit_breakers[component_name] = CircuitBreaker(**kwargs)
            return self.circuit_breakers[component_name]

    def register_retry_policy(self, component_name: str, **kwargs) -> RetryPolicy:
        """Register a retry policy for component."""
        with self.lock:
            if component_name not in self.retry_policies:
                self.retry_policies[component_name] = RetryPolicy(**kwargs)
            return self.retry_policies[component_name]

    def register_fallback(self, component_name: str, operation: str, fallback_func: Callable):
        """Register fallback function."""
        self.fallback_registry.register_fallback(component_name, operation, fallback_func)

    def register_global_error_handler(self, handler: Callable):
        """Register global error handler."""
        with self.lock:
            self.global_error_handlers.append(handler)

    def handle_error(
        self,
        component_name: str,
        operation: str,
        error: Exception,
        error_type: ErrorType = None,
        **kwargs,
    ) -> bool:
        """
        Handle error with appropriate recovery strategies.

        Args:
            component_name: Name of the component
            operation: Operation that failed
            error: The exception that occurred
            error_type: Type of error
            **kwargs: Additional context

        Returns:
            bool: True if error was handled, False if unrecoverable
        """
        with self.lock:
            # Create error context
            context = self.context_manager.create_context(
                component_name,
                operation,
                error_type or ErrorType.VALIDATION_ERROR,
                error=str(error),
                **kwargs,
            )

            # Log error
            logger.error(f"Error in {component_name}.{operation}: {error}")

            # Record circuit breaker failure
            circuit_breaker = self.circuit_breakers.get(component_name)
            if circuit_breaker:
                circuit_breaker.record_failure()

            # Check for fallback
            if self.fallback_registry.has_fallback(component_name, operation):
                fallback_func = self.fallback_registry.get_fallback(component_name, operation)
                try:
                    fallback_func(**context.error_details)
                    context.fallback_attempted = True
                    logger.info(f"Successfully used fallback for {component_name}.{operation}")
                    return True
                except Exception as fallback_error:
                    logger.error(
                        f"Fallback failed for {component_name}.{operation}: {fallback_error}"
                    )
                    context.error_details["fallback_error"] = str(fallback_error)

            # Trigger error callbacks
            self.context_manager.trigger_error_callbacks(component_name, context)

            # Trigger global error handlers
            for handler in self.global_error_handlers:
                try:
                    handler(component_name, operation, error, context)
                except Exception as handler_error:
                    logger.error(f"Global error handler failed: {handler_error}")

            return False

    def execute_with_error_handling(
        self, func: Callable, component_name: str, operation: str, *args, **kwargs
    ) -> Any:
        """
        Execute function with comprehensive error handling.

        Args:
            func: Function to execute
            component_name: Name of the component
            operation: Operation name
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result or None if failed
        """
        circuit_breaker = self.circuit_breakers.get(component_name)
        if circuit_breaker and not circuit_breaker.allow_request():
            self.handle_error(
                component_name,
                operation,
                Exception("Circuit breaker open"),
                error_type=ErrorType.CIRCUIT_BREAKER,
            )
            return None

        retry_policy = self.retry_policies.get(component_name)

        for attempt in range((retry_policy.max_retries if retry_policy else 0) + 1):
            try:
                result = func(*args, **kwargs)

                # Record success if this was a retry
                if attempt > 0:
                    logger.info(
                        f"Operation {component_name}.{operation} succeeded on attempt {attempt + 1}"
                    )
                    if circuit_breaker:
                        circuit_breaker.record_success()

                return result

            except Exception as error:
                # Handle non-retryable errors immediately
                if not self._is_retryable(error):
                    self.handle_error(component_name, operation, error)
                    return None

                # Calculate delay for retry
                delay = retry_policy.get_delay(attempt) if retry_policy else 0.1
                logger.warning(
                    f"Attempt {attempt + 1} failed for {component_name}.{operation}, "
                    f"retrying in {delay:.2f}s: {error}"
                )

                if attempt < retry_policy.max_retries if retry_policy else 0:
                    time.sleep(delay)
                else:
                    self.handle_error(component_name, operation, error)
                    return None

    def _is_retryable(self, error: Exception) -> bool:
        """Check if error is retryable."""
        retryable_errors = (TimeoutError, ConnectionError, OSError, RuntimeError, IOError)
        return isinstance(error, retryable_errors)

    @contextmanager
    def error_context(self, component_name: str, operation: str, error_type: ErrorType = None):
        """Context manager for error handling."""
        self.context_manager.create_context(
            component_name, operation, error_type or ErrorType.VALIDATION_ERROR
        )

        try:
            yield
        except Exception as error:
            self.handle_error(component_name, operation, error, error_type)
            raise
        finally:
            # Clean up context if successful
            if not hasattr(error, "__traceback__") or error.__traceback__ is None:
                circuit_breaker = self.circuit_breakers.get(component_name)
                if circuit_breaker:
                    circuit_breaker.record_success()


# Global error handler instance
error_handler = ErrorHandler()


def with_error_handling(component_name: str, operation: str, error_type: ErrorType = None):
    """Decorator for adding error handling to functions."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return error_handler.execute_with_error_handling(
                func, component_name, operation, *args, **kwargs
            )

        return wrapper

    return decorator


def safe_execute(
    func: Callable,
    default_value: Any = None,
    component_name: str = "unknown",
    operation: str = "unknown",
) -> Any:
    """
    Safely execute a function with error handling.

    Args:
        func: Function to execute
        default_value: Value to return on error
        component_name: Component name for error logging
        operation: Operation name for error logging

    Returns:
        Function result or default_value on error
    """
    try:
        return func()
    except Exception as error:
        logger.error(f"Error in {component_name}.{operation}: {error}")
        error_handler.handle_error(component_name, operation, error)
        return default_value
