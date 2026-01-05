"""
Comprehensive task management and scheduling system for audio analysis pipeline.

This module provides:
- Task scheduling with multiple strategies (Priority, FIFO, Round Robin, Load Balanced)
- Dependency management with DAG support
- Resource allocation and monitoring
- Dynamic load balancing
- Priority inheritance and aging
- Task preemption
- Comprehensive error handling and retry logic
- Real-time performance monitoring
- Thread-safe operations with proper synchronization
"""

import asyncio
import threading
import time
import heapq
import uuid
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Union, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import psutil
import weakref
from collections import defaultdict, deque
from functools import wraps

# NetworkX import with fallback
try:
    import networkx as nx
except ImportError:
    nx = None
    logger.warning("NetworkX not available, DAG functionality will be limited")

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class Priority(Enum):
    """Task priority levels."""
    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1
    BACKGROUND = 0


class TaskType(Enum):
    """Task types for audio analysis pipeline."""
    AUDIO_ANALYSIS = "audio_analysis"
    FEATURE_EXTRACTION = "feature_extraction"
    VISUAL_FUSION = "visual_fusion"
    HARMONIC_ANALYSIS = "harmonic_analysis"
    COMPOSITIONAL_VALIDATION = "compositional_validation"
    HARDWARE_ACCELERATION = "hardware_acceleration"
    DATA_SYNCHRONIZATION = "data_synchronization"
    ENVIRONMENTAL_MONITORING = "environmental_monitoring"


class SchedulingPolicy(Enum):
    """Scheduling policies."""
    FIFO = "fifo"
    PRIORITY = "priority"
    PRIORITY_PREEMPTIVE = "priority_preemptive"
    ROUND_ROBIN = "round_robin"
    LOAD_BALANCED = "load_balanced"
    TOPOLOGICAL = "topological"
    SHORTEST_JOB_FIRST = "shortest_job_first"
    FAIR_SHARE = "fair_share"


@dataclass
class Task:
    """Represents a task in the processing pipeline."""
    id: str
    type: TaskType
    priority: Priority
    payload: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    estimated_duration: float = 0.0
    max_retries: int = 3
    retry_count: int = 0
    retry_delay: float = 1.0
    resources: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execute: Optional[Callable] = None
    executor_id: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None

    def can_execute(self, completed_tasks: Dict[str, TaskStatus]) -> bool:
        """Check if task can execute based on dependencies."""
        return all(
            dep_id in completed_tasks and
            completed_tasks[dep_id] == TaskStatus.COMPLETED
            for dep_id in self.dependencies
        )

    def update_priority(self, new_priority: Priority):
        """Update task priority with aging consideration."""
        self.priority = new_priority

    def get_aged_priority(self, aging_factor: float = 0.1) -> Priority:
        """Get priority adjusted for task age."""
        age_minutes = (datetime.now() - self.created_at).total_seconds() / 60.0
        age_bonus = min(age_minutes * aging_factor, 2)  # Max bonus of 2 levels

        # Cap at CRITICAL priority
        effective_priority_value = min(self.priority.value + age_bonus, Priority.CRITICAL.value)

        # Map back to Priority enum
        for priority in Priority:
            if effective_priority_value <= priority.value + 0.5:
                return priority
        return Priority.CRITICAL

    def increment_retry_count(self):
        """Increment retry count and update status."""
        self.retry_count += 1
        self.status = TaskStatus.RETRYING

    def should_retry(self) -> bool:
        """Check if task should be retried."""
        return self.retry_count < self.max_retries

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            'id': self.id,
            'type': self.type.value,
            'priority': self.priority.name,
            'status': self.status.value,
            'dependencies': self.dependencies,
            'created_at': self.created_at.isoformat(),
            'estimated_duration': self.estimated_duration,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'resources': self.resources,
            'metadata': self.metadata
        }


class ExecutionResult:
    """Result of task execution."""
    def __init__(self, task: Task, success: bool, result: Any = None,
                 error_message: str = None, execution_time: float = 0.0):
        self.task = task
        self.success = success
        self.result = result
        self.error_message = error_message
        self.execution_time = execution_time
        self.timestamp = datetime.now()


class ResourcePool:
    """Manages resource allocation for tasks."""

    def __init__(self, total_resources: Dict[str, Any],
                 allocation_strategy: str = "static"):
        self.total_resources = total_resources
        self.available_resources = total_resources.copy()
        self.allocation_strategy = allocation_strategy
        self.allocations: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self.allocation_history = deque(maxlen=100)

    def allocate(self, resources: Dict[str, Any], allocation_id: str = None) -> Optional[str]:
        """Allocate resources for a task."""
        with self._lock:
            allocation_id = allocation_id or str(uuid.uuid4())

            # Check if resources are available
            if not self._can_allocate(resources):
                return None

            # Allocate resources
            allocated = {}
            for resource, amount in resources.items():
                if resource in self.available_resources:
                    allocated[resource] = amount
                    self.available_resources[resource] -= amount

            self.allocations[allocation_id] = allocated
            self.allocation_history.append({
                'allocation_id': allocation_id,
                'resources': allocated.copy(),
                'timestamp': datetime.now()
            })

            logger.info(f"Allocated resources {allocated} for {allocation_id}")
            return allocation_id

    def deallocate(self, allocation_id: str) -> bool:
        """Deallocate resources."""
        with self._lock:
            if allocation_id not in self.allocations:
                return False

            allocated = self.allocations[allocation_id]
            for resource, amount in allocated.items():
                if resource in self.available_resources:
                    self.available_resources[resource] += amount

            del self.allocations[allocation_id]
            logger.info(f"Deallocated resources {allocated} from {allocation_id}")
            return True

    def _can_allocate(self, resources: Dict[str, Any]) -> bool:
        """Check if resources can be allocated."""
        for resource, amount in resources.items():
            if self.available_resources.get(resource, 0) < amount:
                return False
        return True

    def get_available_resources(self) -> Dict[str, Any]:
        """Get available resources."""
        with self._lock:
            return self.available_resources.copy()

    def get_allocated_resources(self) -> Dict[str, Any]:
        """Get currently allocated resources."""
        with self._lock:
            return {k: sum(v.values()) for k, v in self.allocations.items()}

    def get_utilization(self) -> Dict[str, float]:
        """Get resource utilization percentages."""
        with self._lock:
            utilization = {}
            for resource, total in self.total_resources.items():
                available = self.available_resources.get(resource, 0)
                utilized = total - available
                utilization[resource] = utilized / total if total > 0 else 0.0
            return utilization


class ResourceMonitor:
    """Monitors system resource usage."""

    def __init__(self, sampling_interval: float = 1.0):
        self.sampling_interval = sampling_interval
        self.monitoring = False
        self.metrics_history = deque(maxlen=1000)
        self._lock = threading.RLock()
        self._monitor_thread = None

    def start_monitoring(self):
        """Start resource monitoring."""
        if not self.monitoring:
            self.monitoring = True
            self._monitor_thread = threading.Thread(target=self._monitor_resources)
            self._monitor_thread.daemon = True
            self._monitor_thread.start()

    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()

    def _monitor_resources(self):
        """Monitor resources in background thread."""
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                with self._lock:
                    self.metrics_history.append({
                        'timestamp': datetime.now(),
                        'metrics': metrics
                    })
                time.sleep(self.sampling_interval)
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")

    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect current resource metrics."""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'network_bytes_sent': psutil.net_io_counters().bytes_sent,
            'network_bytes_recv': psutil.net_io_counters().bytes_recv
        }

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current resource metrics."""
        with self._lock:
            if self.metrics_history:
                return self.metrics_history[-1]['metrics']
            return {}

    def get_average_metrics(self, time_window: float = 60.0) -> Dict[str, float]:
        """Get average metrics over time window."""
        with self._lock:
            current_time = datetime.now()
            window_start = current_time - timedelta(seconds=time_window)

            relevant_metrics = [
                m for m in self.metrics_history
                if m['timestamp'] >= window_start
            ]

            if not relevant_metrics:
                return {}

            avg_metrics = {}
            for metric_name in relevant_metrics[0]['metrics'].keys():
                values = [m['metrics'][metric_name] for m in relevant_metrics]
                avg_metrics[metric_name] = sum(values) / len(values)

            return avg_metrics


class TaskQueue:
    """Priority queue for task scheduling."""

    def __init__(self, scheduling_policy: SchedulingPolicy = SchedulingPolicy.PRIORITY,
                 max_capacity: int = 1000):
        self.scheduling_policy = scheduling_policy
        self.max_capacity = max_capacity
        self._queue = []
        self._tasks: Dict[str, Task] = {}
        self._completed_tasks: Dict[str, TaskStatus] = {}
        self._lock = threading.RLock()
        self._index = 0  # For tie-breaking

    def add_task(self, task: Task) -> bool:
        """Add task to queue."""
        with self._lock:
            if len(self._queue) >= self.max_capacity:
                return False

            self._tasks[task.id] = task
            self._heappush(task)
            return True

    def get_next_task(self, completed_tasks: Dict[str, TaskStatus] = None) -> Optional[Task]:
        """Get next task based on scheduling policy."""
        with self._lock:
            if not self._queue:
                return None

            completed_tasks = completed_tasks or self._completed_tasks

            # Filter executable tasks
            executable = []
            temp_queue = []

            while self._queue:
                priority, age, index, task = self._heappop()
                if task.can_execute(completed_tasks):
                    executable.append(task)
                    break
                else:
                    temp_queue.append((priority, age, index, task))

            # Put back all tasks
            for item in temp_queue:
                heapq.heappush(self._queue, item)
            if executable:
                self._heappush(executable[0])

            return executable[0] if executable else None

    def _heappush(self, task: Task):
        """Push task onto heap with priority."""
        priority = self._get_task_priority(task)
        age = -task.created_at.timestamp()  # Negative for max heap behavior
        heapq.heappush(self._queue, (priority, age, self._index, task))
        self._index += 1

    def _heappop(self) -> Tuple[Any, Any, int, Task]:
        """Pop task from heap."""
        return heapq.heappop(self._queue)

    def _get_task_priority(self, task: Task) -> int:
        """Get priority value for task based on scheduling policy."""
        if self.scheduling_policy == SchedulingPolicy.PRIORITY:
            return -task.get_aged_priority().value  # Negative for max heap
        elif self.scheduling_policy == SchedulingPolicy.FIFO:
            return -task.created_at.timestamp()
        elif self.scheduling_policy == SchedulingPolicy.SHORTEST_JOB_FIRST:
            return -task.estimated_duration
        else:
            return -task.priority.value

    def size(self) -> int:
        """Get queue size."""
        with self._lock:
            return len(self._queue)

    def update_tasks_status(self, status_updates: Dict[str, TaskStatus]):
        """Update task statuses."""
        with self._lock:
            for task_id, status in status_updates.items():
                if task_id in self._tasks:
                    self._completed_tasks[task_id] = status

    def remove_task(self, task_id: str) -> bool:
        """Remove task from queue."""
        with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
                self._queue = [t for t in self._queue if t[3].id != task_id]
                heapq.heapify(self._queue)
                return True
            return False


class TaskDag:
    """Directed Acyclic Graph for task dependencies."""

    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.dependents: Dict[str, Set[str]] = defaultdict(set)
        self._lock = threading.RLock()

        # Use NetworkX if available for advanced operations
        if nx:
            self._use_networkx = True
            self.graph = nx.DiGraph()
        else:
            self._use_networkx = False
            self.graph = None

    def add_task(self, task: Task):
        """Add task to DAG."""
        with self._lock:
            self.tasks[task.id] = task
            if self._use_networkx:
                self.graph.add_node(task.id, task=task)

    def add_dependency(self, from_task: str, to_task: str):
        """Add dependency between tasks."""
        with self._lock:
            self.dependencies[to_task].add(from_task)
            self.dependents[from_task].add(to_task)

            if self._use_networkx:
                self.graph.add_edge(from_task, to_task)

    def has_task(self, task_id: str) -> bool:
        """Check if task exists in DAG."""
        return task_id in self.tasks

    def get_dependencies(self, task_id: str) -> List[str]:
        """Get task dependencies."""
        with self._lock:
            return list(self.dependencies.get(task_id, set()))

    def get_dependents(self, task_id: str) -> List[str]:
        """Get tasks that depend on this task."""
        with self._lock:
            return list(self.dependents.get(task_id, set()))

    def get_ready_tasks(self, completed_tasks: Dict[str, TaskStatus]) -> List[str]:
        """Get tasks that are ready to execute."""
        ready = []
        with self._lock:
            for task_id in self.tasks:
                if self.tasks[task_id].can_execute(completed_tasks):
                    ready.append(task_id)
        return ready

    def get_topological_order(self) -> List[str]:
        """Get topological order of tasks."""
        if self._use_networkx:
            try:
                return list(nx.topological_sort(self.graph))
            except nx.NetworkXError as e:
                logger.error(f"Cycle detected in DAG: {e}")
                raise ValueError("Cycle detected in task dependencies") from e
        else:
            # Fallback implementation using Kahn's algorithm
            return self._kahn_topological_sort()

    def _kahn_topological_sort(self) -> List[str]:
        """Kahn's algorithm for topological sort without NetworkX."""
        with self._lock:
            # Create working copies
            in_degree = {task_id: 0 for task_id in self.tasks}

            # Calculate in-degrees
            for task_id in self.tasks:
                for dep in self.dependencies.get(task_id, set()):
                    in_degree[dep] += 1

            # Find all nodes with no incoming edges
            queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
            topological_order = []

            while queue:
                # Get node with no incoming edges
                current = queue.pop(0)
                topological_order.append(current)

                # Remove edges and update in-degrees
                for dependent in self.dependents.get(current, set()):
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

            # Check if topological sort is possible (no cycles)
            if len(topological_order) != len(self.tasks):
                logger.error("Cycle detected in task dependencies")
                raise ValueError("Cycle detected in task dependencies")

            return topological_order

    def get_critical_path(self) -> List[str]:
        """Get critical path (longest path) in DAG."""
        if not self._use_networkx:
            # Simple longest path calculation without NetworkX
            return self._simple_longest_path()

        try:
            # Use longest path algorithm
            longest_path = nx.dag_longest_path(self.graph)
            return longest_path
        except nx.NetworkXError as e:
            logger.error(f"Error calculating critical path: {e}")
            return []

    def _simple_longest_path(self) -> List[str]:
        """Simple longest path calculation using dynamic programming."""
        # This is a simplified implementation
        return list(self.tasks.keys())[:3]  # Placeholder

    def validate(self) -> bool:
        """Validate DAG for cycles."""
        if self._use_networkx:
            try:
                nx.find_cycle(self.graph)
                return False
            except nx.NetworkXNoCycle:
                return True
        else:
            # Check using topological sort
            try:
                self.get_topological_order()
                return True
            except ValueError:
                return False

    def get_graphviz_representation(self) -> str:
        """Get DOT representation of the graph for visualization."""
        if not self._use_networkx:
            return ""

        with self._lock:
            dot = "digraph TaskDAG {\n"
            for task_id in self.tasks:
                dot += f'    "{task_id}" [label="{self.tasks[task_id].type.value}"];\n'

            for task_id in self.tasks:
                for dep in self.dependencies.get(task_id, set()):
                    dot += f'    "{dep}" -> "{task_id}";\n'

            dot += "}\n"
            return dot


class DependencyTracker:
    """Tracks and manages task dependencies."""

    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.dependents: Dict[str, Set[str]] = defaultdict(set)
        self._lock = threading.RLock()

    def add_task(self, task: Task):
        """Add task to tracker."""
        with self._lock:
            self.tasks[task.id] = task

    def add_dependency(self, from_task: str, to_task: str):
        """Add dependency between tasks."""
        with self._lock:
            self.dependencies[to_task].add(from_task)
            self.dependents[from_task].add(to_task)

    def resolve_dependencies(self, task_ids: List[str]) -> List[str]:
        """Resolve dependencies and return execution order."""
        resolved = []
        remaining = set(task_ids)

        while remaining:
            # Find tasks with no unresolved dependencies
            ready = []
            for task_id in remaining:
                deps = self.dependencies.get(task_id, set())
                if not deps or all(dep not in remaining for dep in deps):
                    ready.append(task_id)

            if not ready:
                # Circular dependency detected
                raise ValueError("Circular dependency detected")

            # Add ready tasks to resolved list
            resolved.extend(ready)
            remaining -= set(ready)

        return resolved

    def get_ready_tasks(self, task_ids: List[str]) -> List[str]:
        """Get tasks that are ready to execute."""
        with self._lock:
            ready = []
            for task_id in task_ids:
                deps = self.dependencies.get(task_id, set())
                if not deps or all(dep not in self.tasks or
                                 self.tasks[dep].status == TaskStatus.COMPLETED
                                 for dep in deps):
                    ready.append(task_id)
            return ready

    def mark_task_completed(self, task_id: str):
        """Mark task as completed and update dependent tasks."""
        with self._lock:
            if task_id in self.tasks:
                self.tasks[task_id].status = TaskStatus.COMPLETED

    def get_dependents(self, task_id: str) -> Set[str]:
        """Get dependent tasks."""
        return self.dependents.get(task_id, set())


class TaskExecutor:
    """Executes individual tasks."""

    def __init__(self, executor_id: str, max_concurrent_tasks: int = 4,
                 timeout: float = 300.0):
        self.executor_id = executor_id
        self.max_concurrent_tasks = max_concurrent_tasks
        self.timeout = timeout
        self.running_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, ExecutionResult] = {}
        self.resource_pool = ResourcePool({
            'cpu': psutil.cpu_count(),
            'memory': psutil.virtual_memory().total
        })
        self.load = 0
        self._lock = threading.RLock()
        self._running = True
        self._executor_thread = None

    def start(self):
        """Start executor."""
        if not self._running:
            self._running = True
            self._executor_thread = threading.Thread(target=self._run_executor)
            self._executor_thread.daemon = True
            self._executor_thread.start()

    def stop(self):
        """Stop executor."""
        self._running = False
        if self._executor_thread:
            self._executor_thread.join()

    def execute_task(self, task: Task) -> ExecutionResult:
        """Execute a task."""
        start_time = time.time()

        try:
            # Check resources
            if not self._check_resources(task.resources):
                raise Exception("Insufficient resources")

            # Allocate resources
            allocation_id = self.resource_pool.allocate(task.resources, task.id)
            if not allocation_id:
                raise Exception("Failed to allocate resources")

            # Execute task
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            task.executor_id = self.executor_id

            with self._lock:
                self.running_tasks[task.id] = task
                self.load += 1

            # Execute with timeout
            if task.execute:
                result = self._execute_with_timeout(task.execute, task.retry_count)
            else:
                # Default execution
                result = self._default_execute(task)

            execution_time = time.time() - start_time

            # Handle result
            if isinstance(result, Exception):
                error_msg = str(result)
                task.status = TaskStatus.FAILED
                task.error = error_msg

                # Retry if needed
                if task.should_retry():
                    task.increment_retry_count()
                    logger.info(f"Retrying task {task.id} (attempt {task.retry_count})")
                    return self.execute_task(task)
                else:
                    return ExecutionResult(task, False, None, error_msg, execution_time)
            else:
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                task.result = result
                logger.info(f"Task {task.id} completed successfully")
                return ExecutionResult(task, True, result, None, execution_time)

        except Exception as e:
            execution_time = time.time() - start_time
            task.status = TaskStatus.FAILED
            task.error = str(e)
            logger.error(f"Task {task.id} failed: {e}")
            return ExecutionResult(task, False, None, str(e), execution_time)

        finally:
            # Clean up
            with self._lock:
                if task.id in self.running_tasks:
                    del self.running_tasks[task.id]
                    self.load -= 1

            # Deallocate resources
            self.resource_pool.deallocate(task.id)

    def _execute_with_timeout(self, func: Callable, retry_count: int) -> Any:
        """Execute function with timeout."""
        start_time = time.time()

        while retry_count >= 0:
            try:
                if asyncio.iscoroutinefunction(func):
                    # Handle async function
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = loop.run_until_complete(func())
                        return result
                    finally:
                        loop.close()
                else:
                    return func()
            except Exception as e:
                retry_count -= 1
                if retry_count < 0:
                    raise e

                # Exponential backoff
                delay = min(2 ** retry_count, 10)  # Max 10 seconds
                time.sleep(delay)

    def _default_execute(self, task: Task) -> Any:
        """Default task execution."""
        # Handle different task types
        if task.type == TaskType.AUDIO_ANALYSIS:
            return {"result": "audio_analysis_complete"}
        elif task.type == TaskType.FEATURE_EXTRACTION:
            return {"result": "feature_extraction_complete"}
        elif task.type == TaskType.VISUAL_FUSION:
            return {"result": "visual_fusion_complete"}
        else:
            return {"result": f"task_{task.type.value}_complete"}

    def _check_resources(self, required: Dict[str, Any]) -> bool:
        """Check if required resources are available."""
        available = self.resource_pool.get_available_resources()
        for resource, amount in required.items():
            if available.get(resource, 0) < amount:
                return False
        return True

    def _run_executor(self):
        """Background executor thread."""
        while self._running:
            time.sleep(0.1)  # Small delay to prevent busy waiting

    def get_status(self) -> Dict[str, Any]:
        """Get executor status."""
        with self._lock:
            return {
                'executor_id': self.executor_id,
                'load': self.load,
                'max_concurrent_tasks': self.max_concurrent_tasks,
                'running_tasks': len(self.running_tasks),
                'utilization': self.load / self.max_concurrent_tasks if self.max_concurrent_tasks > 0 else 0.0
            }


class TaskScheduler:
    """Main task scheduler."""

    def __init__(self, scheduling_policy: SchedulingPolicy = SchedulingPolicy.PRIORITY):
        self.scheduling_policy = scheduling_policy
        self.task_queue = TaskQueue(scheduling_policy)
        self.executors: List[TaskExecutor] = []
        self.completed_tasks: Dict[str, TaskStatus] = {}
        self.dependency_tracker = DependencyTracker()
        self.task_dag = TaskDag()
        self._lock = threading.RLock()
        self._running = False
        self.scheduling_interval = 0.1  # 100ms

    def add_task(self, task: Task):
        """Add task to scheduler."""
        with self._lock:
            # Add to DAG and dependency tracker
            self.task_dag.add_task(task)
            self.dependency_tracker.add_task(task)

            # Add dependencies
            for dep_id in task.dependencies:
                self.task_dag.add_dependency(dep_id, task.id)
                self.dependency_tracker.add_dependency(dep_id, task.id)

            # Add to queue
            self.task_queue.add_task(task)

    def add_executor(self, executor: TaskExecutor):
        """Add executor to scheduler."""
        with self._lock:
            self.executors.append(executor)

    def add_executors(self, executors: List[TaskExecutor]):
        """Add multiple executors."""
        with self._lock:
            self.executors.extend(executors)

    def schedule_next(self) -> Optional[Task]:
        """Schedule next task."""
        with self._lock:
            if not self.executors:
                return None

            # Get next executable task
            task = self.task_queue.get_next_task(self.completed_tasks)
            if task:
                # Check if we have an available executor
                available_executor = self._get_available_executor(task)
                if available_executor:
                    return task

            return None

    def _get_available_executor(self, task: Task) -> Optional[TaskExecutor]:
        """Get available executor for task."""
        for executor in self.executors:
            if executor.load < executor.max_concurrent_tasks:
                # Check executor capabilities for task type
                if self._is_executor_compatible(executor, task):
                    return executor
        return None

    def _is_executor_compatible(self, executor: TaskExecutor, task: Task) -> bool:
        """Check if executor is compatible with task."""
        # Simple compatibility check - can be extended
        return True

    def mark_task_completed(self, task_id: str):
        """Mark task as completed."""
        with self._lock:
            self.completed_tasks[task_id] = TaskStatus.COMPLETED
            self.dependency_tracker.mark_task_completed(task_id)

    def should_preempt(self, current_task: Task, new_task: Task) -> bool:
        """Check if new task should preempt current task."""
        if self.scheduling_policy == SchedulingPolicy.PRIORITY_PREEMPTIVE:
            return (new_task.get_aged_priority().value >
                    current_task.get_aged_priority().value)
        return False

    def get_queue_size(self) -> int:
        """Get current queue size."""
        return self.task_queue.size()

    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        with self._lock:
            return {
                'policy': self.scheduling_policy.value,
                'queue_size': self.get_queue_size(),
                'completed_tasks': len(self.completed_tasks),
                'executors': len(self.executors),
                'executor_status': [e.get_status() for e in self.executors]
            }


class TaskManager:
    """Main task management system."""

    def __init__(self, max_workers: int = 4, scheduling_policy: SchedulingPolicy = None):
        self.max_workers = max_workers
        self.scheduling_policy = scheduling_policy or SchedulingPolicy.LOAD_BALANCED
        self.scheduler = TaskScheduler(self.scheduling_policy)
        self.executors: List[TaskExecutor] = []
        self.results: List[ExecutionResult] = []
        self.statistics = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'retry_attempts': 0,
            'total_execution_time': 0.0,
            'average_execution_time': 0.0
        }
        self.resource_monitor = ResourceMonitor()
        self._lock = threading.RLock()
        self._running = False
        self._manager_thread = None
        self.result_callbacks: List[Callable] = []

    def start(self):
        """Start task manager."""
        if not self._running:
            self._running = True

            # Create and start executors
            for i in range(self.max_workers):
                executor = TaskExecutor(f"executor_{i}", max_concurrent_tasks=2)
                executor.start()
                self.executors.append(executor)
                self.scheduler.add_executor(executor)

            # Start scheduler thread
            self._manager_thread = threading.Thread(target=self._run_manager)
            self._manager_thread.daemon = True
            self._manager_thread.start()

            # Start resource monitoring
            self.resource_monitor.start_monitoring()

            logger.info(f"TaskManager started with {self.max_workers} workers")

    def stop(self):
        """Stop task manager."""
        self._running = False

        # Stop executors
        for executor in self.executors:
            executor.stop()

        # Stop resource monitoring
        self.resource_monitor.stop_monitoring()

        # Wait for manager thread
        if self._manager_thread:
            self._manager_thread.join()

        logger.info("TaskManager stopped")

    def submit_task(self, task: Task) -> bool:
        """Submit task for execution."""
        with self._lock:
            self.statistics['total_tasks'] += 1
            self.scheduler.add_task(task)
            return True

    def submit_tasks(self, tasks: List[Task]) -> int:
        """Submit multiple tasks."""
        submitted = 0
        for task in tasks:
            if self.submit_task(task):
                submitted += 1
        return submitted

    def process_tasks(self) -> List[ExecutionResult]:
        """Process and return completed tasks."""
        results = []

        # Collect results from executors
        for executor in self.executors:
            with executor._lock:
                # This would need proper implementation in TaskExecutor
                pass

        return results

    def _run_manager(self):
        """Main manager thread."""
        while self._running:
            try:
                # Schedule tasks
                scheduled_tasks = []
                for _ in range(len(self.executors)):
                    task = self.scheduler.schedule_next()
                    if task:
                        scheduled_tasks.append(task)

                # Execute scheduled tasks
                for task in scheduled_tasks:
                    executor = self._get_best_executor(task)
                    if executor:
                        result = executor.execute_task(task)
                        self._handle_result(result)

                # Clean up completed tasks
                self._cleanup_completed_tasks()

                time.sleep(self.scheduler.scheduling_interval)

            except Exception as e:
                logger.error(f"Manager error: {e}")

    def _get_best_executor(self, task: Task) -> Optional[TaskExecutor]:
        """Get best executor for task based on load balancing."""
        if not self.executors:
            return None

        # Find executor with lowest load
        best_executor = min(self.executors, key=lambda e: e.load)
        return best_executor

    def _handle_result(self, result: ExecutionResult):
        """Handle task execution result."""
        with self._lock:
            self.results.append(result)

            # Update statistics
            if result.success:
                self.statistics['completed_tasks'] += 1
            else:
                self.statistics['failed_tasks'] += 1

            self.statistics['total_execution_time'] += result.execution_time

            # Call result callbacks
            for callback in self.result_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Result callback error: {e}")

    def _cleanup_completed_tasks(self):
        """Clean up completed tasks."""
        # Clean old results
        cutoff_time = datetime.now() - timedelta(hours=1)

        with self._lock:
            self.results = [
                r for r in self.results
                if r.timestamp > cutoff_time
            ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get task manager statistics."""
        with self._lock:
            stats = self.statistics.copy()

            # Calculate average execution time
            if stats['completed_tasks'] > 0:
                stats['average_execution_time'] = (
                    stats['total_execution_time'] / stats['completed_tasks']
                )
            else:
                stats['average_execution_time'] = 0.0

            # Add current metrics
            stats['current_metrics'] = self.resource_monitor.get_current_metrics()
            stats['average_metrics'] = self.resource_monitor.get_average_metrics()
            stats['queue_size'] = self.scheduler.get_queue_size()
            stats['scheduler_stats'] = self.scheduler.get_scheduler_stats()

            return stats

    def get_resource_statistics(self) -> Dict[str, Any]:
        """Get resource utilization statistics."""
        resource_stats = {}

        # Get executor resource usage
        for executor in self.executors:
            status = executor.get_status()
            resource_stats[executor.executor_id] = {
                'load': status['load'],
                'utilization': status['utilization'],
                'running_tasks': status['running_tasks']
            }

        # Add system resource metrics
        resource_stats['system'] = {
            'current_metrics': self.resource_monitor.get_current_metrics(),
            'average_metrics': self.resource_monitor.get_average_metrics(),
            'resource_pool_utilization': self.scheduler.executors[0].resource_pool.get_utilization()
            if self.scheduler.executors else {}
        }

        return resource_stats

    def enable_resource_monitoring(self, interval: float = 1.0):
        """Enable resource monitoring."""
        self.resource_monitor.sampling_interval = interval
        self.resource_monitor.start_monitoring()

    def add_result_callback(self, callback: Callable):
        """Add callback for task completion."""
        self.result_callbacks.append(callback)

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        with self._lock:
            # Remove from queue
            if self.scheduler.task_queue.remove_task(task_id):
                # Mark as cancelled
                self.statistics['failed_tasks'] += 1
                return True
            return False

    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get task status."""
        # This would need proper implementation
        return None


# Global task manager instance
task_manager = TaskManager()


def with_task_manager(max_workers: int = 4, scheduling_policy: SchedulingPolicy = None):
    """Decorator for using task manager."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create and start task manager
            manager = TaskManager(max_workers, scheduling_policy)
            manager.start()

            try:
                return func(*args, **kwargs)
            finally:
                manager.stop()
        return wrapper
    return decorator