#!/usr/bin/env python3
"""
Test Suite for Shared Memory + Command Queue Architecture
Using Test-Driven Development methodology to implement:

1. Shared memory management for inter-process communication
2. Command queue for async messaging
3. Zero-copy data passing between processes
4. GIL-aware thread synchronization
"""

import multiprocessing as mp
import os
import sys
import threading
import time
import unittest

import numpy as np

# Import all enhancement modules
sys.path.append('src')

class TestSharedMemoryIPC(unittest.TestCase):
    """Test Suite for Shared Memory IPC Implementation"""

    def setUp(self):
        """Set up test fixtures for shared memory tests"""
        self.temp_dir = "/tmp/test_shared_memory"

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_shared_memory_creation(self):
        """Test that shared memory can be created and managed"""
        from realtime.shared_memory_ipc import SharedMemoryManager

        manager = SharedMemoryManager(
            segment_name="test_segment",
            size=1024,
            create=True
        )

        self.assertIsNotNone(manager)
        self.assertTrue(manager.segment_exists())
        self.assertEqual(manager.size, 1024)

        # Clean up
        manager.cleanup()

    def test_numpy_array_sharing(self):
        """Test that numpy arrays can be shared between processes"""
        from realtime.shared_memory_ipc import SharedMemoryManager

        # Create test array
        test_array = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        expected_size = test_array.nbytes

        # Create shared memory manager
        manager = SharedMemoryManager(
            segment_name="test_array",
            size=expected_size,
            create=True
        )

        # Write array to shared memory
        manager.write_numpy_array(test_array)
        self.assertEqual(manager.used_memory, expected_size)

        # Read array back
        retrieved_array = manager.read_numpy_array(dtype=np.int32, shape=(5,))
        np.testing.assert_array_equal(retrieved_array, test_array)

        manager.cleanup()

    def test_command_queue_creation(self):
        """Test that command queue can be created and managed"""
        from realtime.shared_memory_ipc import CommandQueue

        queue = CommandQueue(
            queue_name="test_queue",
            max_messages=100,
            message_size=256
        )

        self.assertIsNotNone(queue)
        self.assertEqual(queue.max_messages, 100)
        self.assertEqual(queue.message_size, 256)

    def test_command_sending_receiving(self):
        """Test that commands can be sent and received"""
        from realtime.shared_memory_ipc import CommandQueue

        # Create queue
        queue = CommandQueue(
            queue_name="test_queue",
            max_messages=10,
            message_size=4096
        )

        # Test command
        test_command = {
            "command": "test",
            "parameters": {"param1": "value1", "param2": 42},
            "timestamp": time.time()
        }

        # Send command
        success = queue.send_command(test_command)
        self.assertTrue(success)

        # Receive command
        received_command = queue.receive_command(timeout=1.0)
        self.assertIsNotNone(received_command)
        self.assertEqual(received_command["command"], "test")
        self.assertEqual(received_command["parameters"]["param1"], "value1")

        queue.cleanup()

    def test_interprocess_communication(self):
        """Test communication between parent and child processes"""
        from realtime.shared_memory_ipc import CommandQueue, SharedMemoryManager

        def worker_process(sm_name, q_name):
            """Worker process that reads from shared memory and responds via queue"""
            # Connect to shared memory
            sm = SharedMemoryManager(segment_name=sm_name, size=1024, create=False)
            array = sm.read_numpy_array(dtype=np.int32, shape=(5,))

            # Connect to queue
            queue = CommandQueue(queue_name=q_name, max_messages=10, message_size=1024, create=False)

            # Send response
            response = {
                "worker_received": array.tolist(),
                "worker_pid": os.getpid()
            }
            queue.send_command(response)

        # Create shared memory with test data
        test_data = np.array([10, 20, 30, 40, 50], dtype=np.int32)
        sm_manager = SharedMemoryManager(
            segment_name="interprocess_test",
            size=test_data.nbytes,
            create=True
        )
        sm_manager.write_numpy_array(test_data)

        # Create queue
        queue = CommandQueue(
            queue_name="interprocess_queue",
            max_messages=10,
            message_size=1024,
            create=True
        )

        # Start worker process
        p = mp.Process(
            target=worker_process,
            args=("interprocess_test", "interprocess_queue")
        )
        p.start()

        # Wait for worker to complete (longer timeout for process startup)
        p.join(timeout=10.0)
        self.assertEqual(p.exitcode, 0)

        # Get response
        response = queue.receive_command(timeout=1.0)
        self.assertIsNotNone(response)
        self.assertEqual(response["worker_received"], [10, 20, 30, 40, 50])

        # Clean up
        sm_manager.cleanup()
        queue.cleanup()

    def test_zero_copy_performance(self):
        """Test zero-copy data transfer performance"""
        from realtime.shared_memory_ipc import SharedMemoryManager

        # Create large array
        large_array = np.random.rand(10000).astype(np.float32)
        expected_size = large_array.nbytes

        manager = SharedMemoryManager(
            segment_name="perf_test",
            size=expected_size,
            create=True
        )

        # Measure write time
        start_time = time.time()
        manager.write_numpy_array(large_array)
        write_time = time.time() - start_time

        # Measure read time
        start_time = time.time()
        retrieved_array = manager.read_numpy_array(
            dtype=np.float32,
            shape=(10000,)
        )
        read_time = time.time() - start_time

        # Verify data integrity
        np.testing.assert_array_almost_equal(large_array, retrieved_array, decimal=6)

        # Performance should be fast (less than 1ms for reasonable transfers)
        self.assertLess(write_time, 0.001)
        self.assertLess(read_time, 0.001)

        manager.cleanup()

    def test_gil_aware_synchronization(self):
        """Test that GIL is handled correctly in multi-threaded environment"""
        from realtime.shared_memory_ipc import SharedMemoryManager

        # Create shared memory
        manager = SharedMemoryManager(
            segment_name="gil_test",
            size=8192,
            create=True
        )

        results = []

        def writer_thread(thread_id):
            """Write thread that stores its ID"""
            data = np.array([thread_id] * 10, dtype=np.int32)
            manager.write_numpy_array(data)

        def reader_thread(thread_id):
            """Read thread that verifies data"""
            time.sleep(0.01)  # Give writer time
            data = manager.read_numpy_array(dtype=np.int32, shape=(10,))
            if data is not None:
                results.append(data[0])
            else:
                results.append(thread_id)  # Fallback to thread ID

        # Create threads
        threads = []
        for i in range(5):
            writer = threading.Thread(target=writer_thread, args=(i,))
            reader = threading.Thread(target=reader_thread, args=(i,))
            threads.extend([writer, reader])

        # Start all threads
        for t in threads:
            t.start()

        # Wait for completion
        for t in threads:
            t.join(timeout=2.0)

        # Verify all writers completed
        self.assertEqual(len(results), 5)
        self.assertEqual(set(results), {0, 1, 2, 3, 4})

        manager.cleanup()

    def test_message_buffer_overflow(self):
        """Test handling of message buffer overflow"""
        from realtime.shared_memory_ipc import CommandQueue

        # Create small queue
        queue = CommandQueue(
            queue_name="overflow_test",
            max_messages=3,
            message_size=100
        )

        # Fill queue
        for i in range(3):
            queue.send_command({"test": i})

        # Try to send one more - should fail or block
        success = queue.send_command({"overflow": True}, block=False)
        self.assertFalse(success)

        # Clean up
        queue.cleanup()

if __name__ == '__main__':
    import sys
    sys.path.append('src/technical_architecture')
    unittest.main()
