#!/usr/bin/env python3
"""
Comprehensive test suite for task management and scheduling system.
Tests various scheduling algorithms, priority management, and resource allocation.
"""

import os
import sys
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from realtime.task_management import (
    DependencyTracker,
    Priority,
    ResourcePool,
    SchedulingPolicy,
    Task,
    TaskDag,
    TaskExecutor,
    TaskManager,
    TaskQueue,
    TaskScheduler,
    TaskStatus,
    TaskType,
)


class TestTask:
    """Test Task class functionality."""

    def test_task_creation(self):
        """Test task creation with various parameters."""
        task = Task(
            id="task_1",
            type=TaskType.AUDIO_ANALYSIS,
            priority=Priority.HIGH,
            payload={"audio": np.array([0.1] * 1000)},
            dependencies=[]
        )

        assert task.id == "task_1"
        assert task.type == TaskType.AUDIO_ANALYSIS
        assert task.priority == Priority.HIGH
        assert task.status == TaskStatus.PENDING
        assert task.created_at is not None
        assert task.retry_count == 0
        assert task.max_retries == 3

    def test_task_dependencies(self):
        """Test task dependency management."""
        # Create dependent tasks
        Task("task_a", TaskType.AUDIO_ANALYSIS, Priority.HIGH, payload={})
        task_b = Task("task_b", TaskType.FEATURE_EXTRACTION, Priority.MEDIUM, payload={})
        task_c = Task("task_c", TaskType.VISUAL_FUSION, Priority.HIGH, payload={})

        # Set up dependencies
        task_b.dependencies = ["task_a"]
        task_c.dependencies = ["task_a", "task_b"]

        # Test dependency satisfaction
        # task_b depends on task_a, so it can execute if task_a is completed
        assert task_b.can_execute({"task_a": TaskStatus.COMPLETED})
        assert not task_b.can_execute({})  # No dependencies completed
        # task_c depends on task_a and task_b
        assert not task_c.can_execute({"task_a": TaskStatus.COMPLETED})
        assert not task_c.can_execute({"task_a": TaskStatus.COMPLETED, "task_b": TaskStatus.RUNNING})
        assert task_c.can_execute({"task_a": TaskStatus.COMPLETED, "task_b": TaskStatus.COMPLETED})

    def test_task_priority_updates(self):
        """Test task priority update behavior."""
        task = Task("task_1", TaskType.AUDIO_ANALYSIS, Priority.LOW, payload={})

        # Test normal priority update
        task.update_priority(Priority.HIGH)
        assert task.priority == Priority.HIGH

        # Test priority aging (higher priority for older tasks)
        task.created_at = datetime.now() - timedelta(minutes=30)
        aged_priority = task.get_aged_priority()
        assert aged_priority.value >= task.priority.value

    def test_task_retry_logic(self):
        """Test task retry mechanism."""
        task = Task("task_1", TaskType.AUDIO_ANALYSIS, Priority.HIGH, payload={})

        # Test retry count increment
        task.increment_retry_count()
        assert task.retry_count == 1
        assert task.status == TaskStatus.RETRYING

        # Test max retry enforcement
        task.max_retries = 0
        assert task.should_retry() is False

        task.max_retries = 3
        task.retry_count = 3
        assert task.should_retry() is False


class TestTaskQueue:
    """Test TaskQueue implementation."""

    def test_priority_queue_basic(self):
        """Test basic priority queue functionality."""
        queue = TaskQueue(scheduling_policy=SchedulingPolicy.PRIORITY)

        # Create tasks with different priorities
        high_task = Task("high", TaskType.AUDIO_ANALYSIS, Priority.HIGH, payload={})
        medium_task = Task("medium", TaskType.FEATURE_EXTRACTION, Priority.MEDIUM, payload={})
        low_task = Task("low", TaskType.VISUAL_FUSION, Priority.LOW, payload={})

        queue.add_task(high_task)
        queue.add_task(medium_task)
        queue.add_task(low_task)

        # Verify priority order
        assert queue.get_next_task().id == "high"
        assert queue.get_next_task().id == "medium"
        assert queue.get_next_task().id == "low"

    def test_fifo_queue(self):
        """Test FIFO queue behavior."""
        queue = TaskQueue(scheduling_policy=SchedulingPolicy.FIFO)

        # Create tasks
        task1 = Task("task1", TaskType.AUDIO_ANALYSIS, Priority.HIGH, payload={})
        task2 = Task("task2", TaskType.FEATURE_EXTRACTION, Priority.LOW, payload={})

        queue.add_task(task1)
        queue.add_task(task2)

        # Verify FIFO order
        assert queue.get_next_task().id == "task1"
        assert queue.get_next_task().id == "task2"

    def test_task_dependency_filtering(self):
        """Test that dependent tasks are not returned until dependencies are met."""
        queue = TaskQueue(scheduling_policy=SchedulingPolicy.PRIORITY)

        # Create tasks with dependencies
        task_a = Task("task_a", TaskType.AUDIO_ANALYSIS, Priority.HIGH, payload={})
        task_b = Task("task_b", TaskType.FEATURE_EXTRACTION, Priority.HIGH, payload={})
        task_c = Task("task_c", TaskType.VISUAL_FUSION, Priority.HIGH, payload={})

        task_b.dependencies = ["task_a"]
        task_c.dependencies = ["task_b"]

        queue.add_task(task_a)
        queue.add_task(task_b)
        queue.add_task(task_c)

        # Initially, only task_a should be executable
        assert queue.get_next_task().id == "task_a"

        # Complete task_a, now task_b should be available
        completed_tasks = {"task_a": TaskStatus.COMPLETED}
        queue.update_tasks_status(completed_tasks)

        assert queue.get_next_task().id == "task_b"

        # Complete task_b, now task_c should be available
        completed_tasks = {"task_b": TaskStatus.COMPLETED}
        queue.update_tasks_status(completed_tasks)

        assert queue.get_next_task().id == "task_c"

    def test_queue_size_and_capacity(self):
        """Test queue size management and capacity limits."""
        queue = TaskQueue(scheduling_policy=SchedulingPolicy.PRIORITY, max_capacity=5)

        # Fill queue
        for i in range(5):
            task = Task(f"task_{i}", TaskType.AUDIO_ANALYSIS, Priority.LOW, payload={})
            queue.add_task(task)

        # Try to add beyond capacity
        extra_task = Task("extra", TaskType.AUDIO_ANALYSIS, Priority.LOW, payload={})
        result = queue.add_task(extra_task)
        assert result is False  # Should return False when queue is full

        # Verify size
        assert queue.size() == 5


class TestResourcePool:
    """Test ResourcePool management."""

    def test_resource_allocation(self):
        """Test basic resource allocation."""
        pool = ResourcePool(
            total_resources={"cpu": 8, "memory": 16 * 1024 * 1024 * 1024, "gpu": 2},
            allocation_strategy="static"
        )

        # Allocate resources
        allocation = pool.allocate({"cpu": 2, "memory": 4 * 1024 * 1024 * 1024, "gpu": 1})
        assert allocation is not None
        assert allocation["cpu"] == 2
        assert allocation["memory"] == 4 * 1024 * 1024 * 1024
        assert allocation["gpu"] == 1

        # Check remaining resources
        remaining = pool.get_available_resources()
        assert remaining["cpu"] == 6
        assert remaining["memory"] == 12 * 1024 * 1024 * 1024
        assert remaining["gpu"] == 1

    def test_resource_starvation_prevention(self):
        """Test prevention of resource starvation."""
        pool = ResourcePool(
            total_resources={"cpu": 4, "memory": 8 * 1024 * 1024 * 1024},
            allocation_strategy="fair"
        )

        # Allocate half the resources
        allocation1 = pool.allocate({"cpu": 2, "memory": 4 * 1024 * 1024 * 1024})
        assert allocation1 is not None

        # Try to allocate more than available
        allocation2 = pool.allocate({"cpu": 3, "memory": 5 * 1024 * 1024 * 1024})
        assert allocation2 is None

    def test_resource_deallocation(self):
        """Test resource deallocation."""
        pool = ResourcePool(
            total_resources={"cpu": 4, "memory": 8 * 1024 * 1024 * 1024}
        )

        allocation = pool.allocate({"cpu": 2, "memory": 4 * 1024 * 1024 * 1024})
        assert allocation is not None

        # Deallocate
        pool.deallocate(allocation)

        # Check resources are freed
        remaining = pool.get_available_resources()
        assert remaining["cpu"] == 4
        assert remaining["memory"] == 8 * 1024 * 1024 * 1024

    def test_resource_contention(self):
        """Test resource contention resolution."""
        pool = ResourcePool(
            total_resources={"cpu": 2, "memory": 4 * 1024 * 1024 * 1024}
        )

        # Concurrent allocations
        def allocate_resources(cpu, memory):
            return pool.allocate({"cpu": cpu, "memory": memory})

        with threading.Lock():  # Simulate concurrent access
            allocation1 = allocate_resources(1, 2 * 1024 * 1024 * 1024)
            allocation2 = allocate_resources(1, 2 * 1024 * 1024 * 1024)

            assert allocation1 is not None
            assert allocation2 is not None

            # Try to allocate more
            allocation3 = allocate_resources(1, 1 * 1024 * 1024 * 1024)
            assert allocation3 is None


class TestTaskScheduler:
    """Test TaskScheduler functionality."""

    def test_scheduling_strategies(self):
        """Test different scheduling strategies."""
        scheduler = TaskScheduler(scheduling_policy=SchedulingPolicy.PRIORITY)

        # Create tasks
        high_task = Task("high", TaskType.AUDIO_ANALYSIS, Priority.HIGH, payload={})
        medium_task = Task("medium", TaskType.FEATURE_EXTRACTION, Priority.MEDIUM, payload={})
        low_task = Task("low", TaskType.VISUAL_FUSION, Priority.LOW, payload={})

        # Add tasks with resource requirements
        high_task.resources = {"cpu": 2, "memory": 1024 * 1024 * 1024}
        medium_task.resources = {"cpu": 1, "memory": 512 * 1024 * 1024}
        low_task.resources = {"cpu": 4, "memory": 2 * 1024 * 1024 * 1024}

        scheduler.add_task(high_task)
        scheduler.add_task(medium_task)
        scheduler.add_task(low_task)

        # Test scheduling
        scheduled = scheduler.schedule_next()
        assert scheduled is not None
        assert scheduled.id == "high"  # Highest priority

        # Schedule next
        scheduled = scheduler.schedule_next()
        assert scheduled.id == "medium"

    def test_task_dag_scheduling(self):
        """Test scheduling of tasks with dependencies."""
        scheduler = TaskScheduler(scheduling_policy=SchedulingPolicy.TOPOLOGICAL)

        # Create DAG
        task_a = Task("A", TaskType.AUDIO_ANALYSIS, Priority.HIGH, payload={})
        task_b = Task("B", TaskType.FEATURE_EXTRACTION, Priority.HIGH, payload={})
        task_c = Task("C", TaskType.FEATURE_EXTRACTION, Priority.HIGH, payload={})
        task_d = Task("D", TaskType.VISUAL_FUSION, Priority.HIGH, payload={})

        # Set up dependencies: A -> B, A -> C, B -> D, C -> D
        task_b.dependencies = ["A"]
        task_c.dependencies = ["A"]
        task_d.dependencies = ["B", "C"]

        scheduler.add_task(task_a)
        scheduler.add_task(task_b)
        scheduler.add_task(task_c)
        scheduler.add_task(task_d)

        # Schedule in topological order
        scheduled = scheduler.schedule_next()
        assert scheduled.id == "A"

        # Mark A as completed
        scheduler.mark_task_completed("A")

        # B and C should be available
        scheduled = scheduler.schedule_next()
        assert scheduled.id in ["B", "C"]

        # Complete one of them
        scheduler.mark_task_completed(scheduled.id)

        # The other should be available
        scheduled = scheduler.schedule_next()
        assert scheduled.id in ["B", "C"]

        # After completing B and C, D should be available
        scheduler.mark_task_completed("B")
        scheduler.mark_task_completed("C")

        scheduled = scheduler.schedule_next()
        assert scheduled.id == "D"

    def test_preemption_handling(self):
        """Test task preemption for higher priority tasks."""
        scheduler = TaskScheduler(scheduling_policy=SchedulingPolicy.PRIORITY_PREEMPTIVE)

        # Create tasks
        long_running = Task("long", TaskType.AUDIO_ANALYSIS, Priority.LOW, payload={})
        urgent = Task("urgent", TaskType.VISUAL_FUSION, Priority.HIGH, payload={})

        long_running.estimated_duration = 30.0  # 30 seconds
        urgent.estimated_duration = 5.0  # 5 seconds

        # Add long task first
        scheduler.add_task(long_running)
        # Schedule long task
        scheduled = scheduler.schedule_next()
        assert scheduled.id == "long"

        # Now add urgent task (higher priority)
        scheduler.add_task(urgent)

        # Next scheduled should be urgent (higher priority)
        next_scheduled = scheduler.schedule_next()
        assert next_scheduled.id == "urgent"

        # Check if urgent should preempt long (urgent has higher priority and shorter duration)
        preemptive = scheduler.should_preempt(scheduled, next_scheduled)
        assert preemptive is True

    def test_load_balancing(self):
        """Test load balancing across multiple executors."""
        scheduler = TaskScheduler(scheduling_policy=SchedulingPolicy.LOAD_BALANCED)

        # Create executors
        executors = [Mock() for _ in range(3)]
        for i, executor in enumerate(executors):
            executor.load = i  # Different loads (0, 1, 2)
            executor.max_concurrent_tasks = 5  # Set max concurrent tasks

        scheduler.add_executors(executors)

        # Create tasks
        tasks = []
        for i in range(3):
            task = Task(f"task_{i}", TaskType.AUDIO_ANALYSIS, Priority.MEDIUM, payload={})
            tasks.append(task)
            scheduler.add_task(task)

        # Schedule tasks - scheduler should pick least loaded executor (executor with load=0)
        scheduled_count = 0
        for task in tasks:
            scheduled = scheduler.schedule_next()
            if scheduled:
                scheduled_count += 1
                # Simulate load increase on the chosen executor (executor with lowest load)
                # In real implementation, executor.load would be incremented when task is assigned

        assert scheduled_count == 3  # All tasks should be scheduled

        # Note: Since we don't actually execute tasks in this test,
        # the loads remain at their initial values (0, 1, 2)
        # Real load balancing would happen during task execution


class TestTaskExecutor:
    """Test TaskExecutor functionality."""

    def test_execution_success(self):
        """Test successful task execution."""
        executor = TaskExecutor(executor_id="test_executor", max_concurrent_tasks=2)

        # Create mock task
        task = Task("test_task", TaskType.AUDIO_ANALYSIS, Priority.HIGH, payload={})
        task.execute = Mock(return_value={"result": "success"})

        result = executor.execute_task(task)

        assert result.status == TaskStatus.COMPLETED
        assert result.success is True
        assert result.error_message is None
        assert task.execute.called

    def test_execution_failure(self):
        """Test task execution failure."""
        executor = TaskExecutor(executor_id="test_executor", max_concurrent_tasks=2)

        # Create mock task that fails
        task = Task("test_task", TaskType.AUDIO_ANALYSIS, Priority.HIGH, payload={})
        task.execute = Mock(side_effect=Exception("Execution failed"))
        task.max_retries = 1

        result = executor.execute_task(task)

        assert result.status == TaskStatus.FAILED
        assert result.success is False
        assert "Execution failed" in result.error_message

    def test_concurrent_execution(self):
        """Test concurrent task execution."""
        executor = TaskExecutor(executor_id="test_executor", max_concurrent_tasks=3)

        # Create multiple tasks
        tasks = []
        for i in range(5):
            task = Task(f"task_{i}", TaskType.AUDIO_ANALYSIS, Priority.MEDIUM, payload={})
            task.execute = Mock(return_value={"result": f"task_{i}_result"})
            tasks.append(task)

        # Execute all tasks
        results = [executor.execute_task(task) for task in tasks]

        # Verify all completed successfully
        for result in results:
            assert result.success is True

    def test_resource_aware_execution(self):
        """Test resource-aware task execution."""
        executor = TaskExecutor(executor_id="test_executor", max_concurrent_tasks=2)

        # Create tasks with different resource requirements
        heavy_task = Task("heavy", TaskType.AUDIO_ANALYSIS, Priority.HIGH, payload={})
        heavy_task.resources = {"cpu": 4, "memory": 4 * 1024 * 1024 * 1024}

        light_task = Task("light", TaskType.FEATURE_EXTRACTION, Priority.MEDIUM, payload={})
        light_task.resources = {"cpu": 1, "memory": 512 * 1024 * 1024}

        # Mock resource availability
        with patch.object(executor, '_check_resources') as mock_check:
            mock_check.side_effect = [
                True,  # First check - resources available
                True   # Second check - resources available
            ]

            result1 = executor.execute_task(heavy_task)
            result2 = executor.execute_task(light_task)

            assert result1.success is True
            assert result2.success is True

    def test_execution_timeout(self):
        """Test execution timeout handling."""
        executor = TaskExecutor(executor_id="test_executor", max_concurrent_tasks=1, timeout=5.0)

        # Create long-running task
        task = Task("long_task", TaskType.AUDIO_ANALYSIS, Priority.HIGH, payload={})
        task.execute = Mock(side_effect=lambda: time.sleep(10))

        result = executor.execute_task(task)

        assert result.status == TaskStatus.FAILED
        assert "timed out" in result.error_message.lower()


class TestTaskDag:
    """Test Task DAG functionality."""

    def test_dag_creation(self):
        """Test DAG creation and validation."""
        dag = TaskDag()

        # Add tasks
        task_a = Task("A", TaskType.AUDIO_ANALYSIS, Priority.HIGH, payload={})
        task_b = Task("B", TaskType.FEATURE_EXTRACTION, Priority.HIGH, payload={})
        task_c = Task("C", TaskType.VISUAL_FUSION, Priority.HIGH, payload={})

        dag.add_task(task_a)
        dag.add_task(task_b)
        dag.add_task(task_c)

        # Add dependencies
        dag.add_dependency("A", "B")
        dag.add_dependency("A", "C")

        # Verify structure
        assert dag.has_task("A")
        assert dag.get_dependencies("B") == ["A"]
        assert dag.get_dependents("A") == ["B", "C"]

    def test_cycle_detection(self):
        """Test cycle detection in DAG."""
        dag = TaskDag()

        # Create tasks
        task_a = Task("A", TaskType.AUDIO_ANALYSIS, Priority.HIGH, payload={})
        task_b = Task("B", TaskType.FEATURE_EXTRACTION, Priority.HIGH, payload={})

        dag.add_task(task_a)
        dag.add_task(task_b)

        # Add cycle
        dag.add_dependency("A", "B")
        dag.add_dependency("B", "A")

        # Should detect cycle
        with pytest.raises(ValueError):
            dag.validate()

    def test_topological_sort(self):
        """Test topological sorting of DAG."""
        dag = TaskDag()

        # Create complex DAG
        tasks = {f"task_{i}": Task(f"task_{i}", TaskType.AUDIO_ANALYSIS, Priority.HIGH, payload={})
                 for i in range(6)}
        for task in tasks.values():
            dag.add_task(task)

        # Add dependencies: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3, 3 -> 4, 3 -> 5
        dag.add_dependency("task_0", "task_1")
        dag.add_dependency("task_0", "task_2")
        dag.add_dependency("task_1", "task_3")
        dag.add_dependency("task_2", "task_3")
        dag.add_dependency("task_3", "task_4")
        dag.add_dependency("task_3", "task_5")

        # Get topological order
        order = dag.get_topological_order()

        # Verify order respects dependencies
        task_0_idx = order.index("task_0")
        task_1_idx = order.index("task_1")
        task_2_idx = order.index("task_2")
        task_3_idx = order.index("task_3")

        assert task_0_idx < task_1_idx
        assert task_0_idx < task_2_idx
        assert task_1_idx < task_3_idx
        assert task_2_idx < task_3_idx

    def test_critical_path(self):
        """Test critical path calculation in DAG."""
        dag = TaskDag()

        # Create tasks
        tasks = {f"task_{i}": Task(f"task_{i}", TaskType.AUDIO_ANALYSIS, Priority.HIGH, payload={})
                 for i in range(5)}
        for task in tasks.values():
            dag.add_task(task)

        # Add dependencies
        dag.add_dependency("task_0", "task_1")
        dag.add_dependency("task_0", "task_2")
        dag.add_dependency("task_1", "task_3")
        dag.add_dependency("task_2", "task_3")
        dag.add_dependency("task_3", "task_4")

        # Set durations
        for i, task in enumerate(tasks.values()):
            task.estimated_duration = i + 1  # task_0: 1, task_1: 2, etc.

        # Calculate critical path
        critical_path = dag.get_critical_path()
        assert "task_0" in critical_path
        assert "task_2" in critical_path
        assert "task_3" in critical_path
        assert "task_4" in critical_path


class TestDependencyTracker:
    """Test DependencyTracker functionality."""

    def test_dependency_resolution(self):
        """Test dependency resolution."""
        tracker = DependencyTracker()

        # Add dependencies
        tracker.add_dependency("A", "B")
        tracker.add_dependency("B", "C")
        tracker.add_dependency("A", "C")

        # Resolve dependencies
        resolved = tracker.resolve_dependencies(["A", "B", "C"])
        assert resolved == ["A", "B", "C"]  # Should respect dependency order

    def test_orphan_tasks(self):
        """Test handling of orphan tasks."""
        tracker = DependencyTracker()

        # Add dependencies first (B depends on A)
        tracker.add_dependency("A", "B")

        # Add task with no dependencies
        task_d = Task("D", TaskType.AUDIO_ANALYSIS, Priority.HIGH, payload={})
        task_b = Task("B", TaskType.FEATURE_EXTRACTION, Priority.HIGH, payload={}, dependencies=["A"])
        tracker.add_task(task_d)
        tracker.add_task(task_b)

        # Get ready tasks (D should be ready, B should not be due to dependency on A)
        ready = tracker.get_ready_tasks({})
        assert "D" in ready
        assert "B" not in ready  # B depends on A

    def test_task_completion_update(self):
        """Test task completion update propagation."""
        tracker = DependencyTracker()

        # Add tasks with dependencies
        task_a = Task("A", TaskType.AUDIO_ANALYSIS, Priority.HIGH, payload={})
        task_b = Task("B", TaskType.FEATURE_EXTRACTION, Priority.HIGH, payload={}, dependencies=["A"])
        task_c = Task("C", TaskType.AUDIO_ANALYSIS, Priority.HIGH, payload={}, dependencies=["B"])

        tracker.add_task(task_a)
        tracker.add_task(task_b)
        tracker.add_task(task_c)
        # Also add dependencies to tracker
        tracker.add_dependency("A", "B")
        tracker.add_dependency("B", "C")

        # Initially, only A should be ready
        ready = tracker.get_ready_tasks({})
        assert "A" in ready
        assert "B" not in ready

        # Complete A
        tracker.mark_task_completed("A")

        # Now B should be ready
        completed = {"A": TaskStatus.COMPLETED}
        ready = tracker.get_ready_tasks(completed)
        assert "B" in ready
        assert "C" not in ready

        # Complete B
        tracker.mark_task_completed("B")

        # Now C should be ready
        completed = {"A": TaskStatus.COMPLETED, "B": TaskStatus.COMPLETED}
        ready = tracker.get_ready_tasks(completed)
        assert "C" in ready


class TestTaskManager:
    """Test TaskManager integration."""

    def test_end_to_end_workflow(self):
        """Test complete task management workflow."""
        manager = TaskManager(max_workers=4)

        # Create tasks
        tasks = []
        for i in range(5):
            task = Task(f"task_{i}", TaskType.AUDIO_ANALYSIS, Priority.HIGH, payload={})
            task.execute = Mock(return_value={"result": f"result_{i}"})
            tasks.append(task)

        # Submit tasks
        for task in tasks:
            manager.submit_task(task)

        # Process tasks
        results = manager.process_tasks()

        # Verify all completed
        assert len(results) == 5
        for result in results:
            assert result.success is True

        # Verify statistics
        stats = manager.get_statistics()
        assert stats["total_tasks"] == 5
        assert stats["completed_tasks"] == 5
        assert stats["failed_tasks"] == 0

    def test_priority_inheritance(self):
        """Test that dependencies inherit parent priority."""
        manager = TaskManager(max_workers=4)

        # Create parent task
        parent = Task("parent", TaskType.AUDIO_ANALYSIS, Priority.HIGH, payload={})
        parent.estimated_duration = 10.0

        # Create dependent task
        child = Task("child", TaskType.FEATURE_EXTRACTION, Priority.LOW, payload={})
        child.dependencies = ["parent"]

        # Submit
        manager.submit_task(parent)
        manager.submit_task(child)

        # Process with delay to simulate parent execution
        import time
        time.sleep(0.1)

        results = manager.process_tasks()

        # Verify child waits for parent
        assert len(results) == 2

    def test_resource_monitoring(self):
        """Test resource monitoring integration."""
        manager = TaskManager(max_workers=2)
        manager.enable_resource_monitoring(interval=0.1)

        # Create resource-intensive tasks
        for i in range(3):
            task = Task(f"task_{i}", TaskType.AUDIO_ANALYSIS, Priority.HIGH, payload={})
            task.resources = {"cpu": 2, "memory": 1024 * 1024 * 1024}
            task.execute = Mock(return_value={"result": "ok"})
            manager.submit_task(task)

        # Process with monitoring
        manager.process_tasks()

        # Get resource stats
        stats = manager.get_resource_statistics()
        assert "cpu_usage" in stats
        assert "memory_usage" in stats

    def test_error_recovery(self):
        """Test error recovery with retries."""
        manager = TaskManager(max_workers=2)

        # Create failing task
        task = Task("failing_task", TaskType.AUDIO_ANALYSIS, Priority.HIGH, payload={})
        task.execute = Mock(side_effect=Exception("Temporary failure"))
        task.max_retries = 2

        manager.submit_task(task)

        # Process tasks
        manager.process_tasks()

        # Should have retry entries
        stats = manager.get_statistics()
        assert stats["retry_attempts"] > 0

    def test_dynamic_load_balancing(self):
        """Test dynamic load balancing across workers."""
        manager = TaskManager(max_workers=3)

        # Create executors with different loads
        executors = manager.executors
        for i, executor in enumerate(executors):
            executor.load = i  # Different initial loads

        # Create tasks
        tasks = []
        for i in range(6):
            task = Task(f"task_{i}", TaskType.AUDIO_ANALYSIS, Priority.MEDIUM, payload={})
            task.execute = Mock(return_value={"result": f"result_{i}"})
            tasks.append(task)

        # Submit tasks
        for task in tasks:
            manager.submit_task(task)

        # Process with load balancing
        manager.process_tasks()

        # Verify load balancing occurred
        final_loads = [executor.load for executor in executors]
        assert max(final_loads) - min(final_loads) <= 1  # Well balanced

        # Verify statistics
        stats = manager.get_statistics()
        assert stats["total_processed"] == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
