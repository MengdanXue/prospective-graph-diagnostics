"""Portable injected-backend checks; no native power API is ever invoked."""
from __future__ import annotations

import importlib
import threading
import unittest
from unittest.mock import patch

from scripts import input_robustness_runtime as runtime


class FakeBackend:
    def __init__(self, failures=None, request_handle=101, registration_handle=202):
        self.calls = []
        self.failures = dict(failures or {})
        self.request_handle = request_handle
        self.registration_handle = registration_handle
        self.callback = None
        self.live_types = set()
        self.closed = False

    def call(self, name, *arguments):
        self.calls.append((name, *arguments))
        key = (name, *arguments)
        failure = self.failures.get(key, self.failures.get(name))
        if failure:
            raise failure

    def create_request(self, reason):
        self.call('create', reason)
        return self.request_handle

    def set_request(self, handle, request_type):
        self.call('set', handle, request_type)
        self.live_types.add(request_type)

    def clear_request(self, handle, request_type):
        self.call('clear', handle, request_type)
        self.live_types.remove(request_type)

    def close_handle(self, handle):
        self.call('close', handle)
        self.live_types.clear()
        self.closed = True

    def register_notification(self, callback):
        self.call('register')
        self.callback = callback
        return self.registration_handle

    def unregister_notification(self, handle):
        self.call('unregister', handle)
        self.callback = None

    def emit(self, code):
        return self.callback(code)


class PowerRequestTests(unittest.TestCase):
    def test_import_and_construction_do_not_load_windows_or_request_power(self):
        with patch('ctypes.WinDLL', create=True, side_effect=AssertionError('native call')):
            importlib.reload(runtime)
            request = runtime.TaskPowerRequest()
            watcher = runtime.PowerEventWatcher()
        self.assertFalse(request.snapshot()['acquired'])
        self.assertFalse(watcher.snapshot()['registered'])

    def test_exact_two_requests_are_released_once_for_normal_context_exit(self):
        backend = FakeBackend()
        request = runtime.TaskPowerRequest(backend)
        with request:
            self.assertEqual(backend.live_types, {1, 3})
            self.assertIs(request.acquire(), request)
            self.assertFalse(request.snapshot()['released'])
        request.release()
        self.assertEqual(backend.calls, [('create', runtime.DEFAULT_REASON),
                                        ('set', 101, 1), ('set', 101, 3),
                                        ('clear', 101, 3), ('clear', 101, 1), ('close', 101)])
        snapshot = request.snapshot()
        self.assertTrue(snapshot['acquired'])
        self.assertTrue(snapshot['released'])
        self.assertTrue(snapshot['handle_closed'])
        self.assertFalse(snapshot['cleanup_errors'])
        self.assertFalse(snapshot['permanent_policy_changed'])
        self.assertEqual(snapshot['active_request_types'], [])
        with self.assertRaisesRegex(RuntimeError, 'cannot be reused'):
            request.acquire()

    def test_all_python_exit_kinds_release_both_requests(self):
        for error in (ValueError('worker failure'), KeyboardInterrupt(), SystemExit(75)):
            with self.subTest(error=type(error).__name__):
                backend = FakeBackend()
                request = runtime.TaskPowerRequest(backend)
                with self.assertRaises(type(error)):
                    with request:
                        raise error
                self.assertTrue(backend.closed)
                self.assertFalse(backend.live_types)
                self.assertTrue(request.snapshot()['released'])

    def test_second_set_failure_clears_first_request_and_closes(self):
        backend = FakeBackend({('set', 101, 3): OSError('execution unavailable')})
        request = runtime.TaskPowerRequest(backend)
        with self.assertRaisesRegex(OSError, 'execution unavailable'):
            request.acquire()
        self.assertFalse(request.snapshot()['acquired'])
        self.assertTrue(request.snapshot()['released'])
        self.assertEqual(backend.calls[-2:], [('clear', 101, 1), ('close', 101)])
        self.assertFalse(backend.live_types)

    def test_first_set_failure_closes_without_clearing_unset_requests(self):
        backend = FakeBackend({('set', 101, 1): OSError('system unavailable')})
        request = runtime.TaskPowerRequest(backend)
        with self.assertRaises(OSError):
            request.acquire()
        self.assertEqual(backend.calls[-1], ('close', 101))
        self.assertFalse(any(call[0] == 'clear' for call in backend.calls))

    def test_failed_creation_never_closes_an_invalid_handle(self):
        for handle in (None, 0, -1):
            with self.subTest(handle=handle):
                backend = FakeBackend(request_handle=handle)
                request = runtime.TaskPowerRequest(backend)
                with self.assertRaisesRegex(RuntimeError, 'invalid handle'):
                    request.acquire()
                self.assertEqual(len(backend.calls), 1)
                self.assertTrue(request.snapshot()['released'])

    def test_clear_failure_still_clears_other_type_and_closes_but_fails_acceptance(self):
        backend = FakeBackend({('clear', 101, 3): OSError('clear failure')})
        request = runtime.TaskPowerRequest(backend).acquire()
        with self.assertRaisesRegex(RuntimeError, 'cleanup failed'):
            request.release()
        self.assertEqual(backend.calls[-3:], [('clear', 101, 3), ('clear', 101, 1), ('close', 101)])
        self.assertTrue(request.snapshot()['released'])
        self.assertEqual(request.snapshot()['cleanup_errors'][0]['request_type'], 3)
        self.assertFalse(backend.live_types)

    def test_failed_close_retains_handle_for_finally_retry_without_double_clear(self):
        backend = FakeBackend({'close': OSError('close failure')})
        request = runtime.TaskPowerRequest(backend).acquire()
        with self.assertRaisesRegex(RuntimeError, 'cleanup failed'):
            request.release()
        self.assertFalse(request.snapshot()['released'])
        self.assertFalse(request.snapshot()['handle_closed'])
        with self.assertRaisesRegex(RuntimeError, 'cannot be reused'):
            request.acquire()
        backend.failures.clear()
        request.release()
        self.assertTrue(request.snapshot()['released'])
        self.assertTrue(request.snapshot()['cleanup_errors'])
        self.assertEqual([call for call in backend.calls if call[0] == 'clear'],
                         [('clear', 101, 3), ('clear', 101, 1)])

    def test_partial_acquisition_preserves_primary_and_cleanup_failures(self):
        backend = FakeBackend({('set', 101, 3): OSError('primary'),
                               ('clear', 101, 1): OSError('cleanup')})
        request = runtime.TaskPowerRequest(backend)
        with self.assertRaisesRegex(OSError, 'primary'):
            request.acquire()
        self.assertTrue(backend.closed)
        self.assertTrue(request.snapshot()['cleanup_errors'])

    def test_snapshot_cannot_mutate_power_evidence(self):
        request = runtime.TaskPowerRequest(FakeBackend()).acquire()
        snapshot = request.snapshot()
        snapshot['events'].clear()
        snapshot['active_request_types'].clear()
        self.assertTrue(request.snapshot()['events'])
        self.assertEqual(request.snapshot()['active_request_types'], [1, 3])
        request.release()

    def test_audit_failure_cannot_bypass_other_clear_or_handle_close(self):
        backend = FakeBackend({('clear', 101, 3): OSError('native clear failure')})
        request = runtime.TaskPowerRequest(backend).acquire()
        with patch.object(runtime, '_event', side_effect=RuntimeError('timestamp unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'cleanup failed'):
                request.release()
        self.assertEqual(backend.calls[-3:], [('clear', 101, 3), ('clear', 101, 1), ('close', 101)])
        self.assertTrue(backend.closed)
        self.assertFalse(backend.live_types)
        self.assertTrue(request.snapshot()['released'])
        self.assertTrue(any(error['operation']=='record_event'
                            for error in request.snapshot()['cleanup_errors']))

    def test_successful_cleanup_with_failed_audit_still_closes_then_reports_failure(self):
        backend = FakeBackend()
        request = runtime.TaskPowerRequest(backend).acquire()
        with patch.object(runtime, '_event', side_effect=KeyboardInterrupt('during audit')):
            with self.assertRaisesRegex(RuntimeError, 'cleanup failed'):
                request.release()
        self.assertTrue(backend.closed)
        self.assertFalse(backend.live_types)
        self.assertTrue(request.snapshot()['cleanup_errors'])

    def test_acquisition_and_failure_logging_errors_still_release_partial_request(self):
        backend = FakeBackend()
        request = runtime.TaskPowerRequest(backend)
        original = runtime._event
        def event(name, **details):
            if name in ('request_set', 'acquire_failed'):
                raise RuntimeError('audit failure')
            return original(name, **details)
        with patch.object(runtime, '_event', side_effect=event):
            with self.assertRaisesRegex(RuntimeError, 'event recording failed'):
                request.acquire()
        self.assertEqual(backend.calls[-2:], [('clear', 101, 1), ('close', 101)])
        self.assertTrue(request.snapshot()['released'])
        self.assertFalse(request.snapshot()['acquired'])
        self.assertEqual(sum(error['operation']=='record_event'
                             for error in request.snapshot()['cleanup_errors']), 2)

    def test_failure_before_first_audit_event_never_acquires_native_resources(self):
        backend = FakeBackend()
        request = runtime.TaskPowerRequest(backend)
        with patch.object(runtime, '_event', side_effect=OSError('clock unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'event recording failed'):
                request.acquire()
        self.assertEqual(backend.calls, [])
        self.assertTrue(request.snapshot()['released'])
        self.assertTrue(request.snapshot()['cleanup_errors'])


class PowerEventTests(unittest.TestCase):
    def test_short_suspend_resume_pair_is_observed_without_polling_or_sleep(self):
        backend = FakeBackend()
        with runtime.PowerEventWatcher(backend) as watcher:
            self.assertEqual(watcher.events, [])
            self.assertIs(watcher.start(), watcher)
            self.assertEqual(backend.emit(4), 0)
            self.assertEqual(backend.emit(18), 0)
            self.assertEqual(backend.emit(7), 0)
            self.assertEqual(backend.emit(999), 0)
            self.assertEqual([event['code'] for event in watcher.events], [4, 18, 7])
            self.assertLess(watcher.events[1]['monotonic_seconds']
                            - watcher.events[0]['monotonic_seconds'], 5)
        self.assertIsNone(backend.callback)
        self.assertTrue(watcher.snapshot()['registered'])
        self.assertTrue(watcher.snapshot()['stopped'])
        self.assertFalse(watcher.snapshot()['subscription_active'])
        watcher.stop()
        self.assertEqual(backend.calls, [('register',), ('unregister', 202)])
        with self.assertRaisesRegex(RuntimeError, 'cannot be reused'):
            watcher.start()

    def test_notification_from_another_thread_is_safe_and_snapshots_are_independent(self):
        backend = FakeBackend()
        with runtime.PowerEventWatcher(backend) as watcher:
            producer = threading.Thread(target=lambda: [backend.emit(4) for _ in range(50)])
            producer.start()
            producer.join(timeout=5)
            self.assertFalse(producer.is_alive())
            self.assertEqual(len(watcher.events), 50)
            snapshot = watcher.events
            snapshot[0]['code'] = 999
            snapshot.clear()
            self.assertEqual(watcher.events[0]['code'], 4)

    def test_exception_and_interrupt_paths_unregister(self):
        for error in (RuntimeError('worker'), KeyboardInterrupt(), SystemExit(75)):
            with self.subTest(error=type(error).__name__):
                backend = FakeBackend()
                watcher = runtime.PowerEventWatcher(backend)
                with self.assertRaises(type(error)):
                    with watcher:
                        raise error
                self.assertIsNone(backend.callback)
                self.assertTrue(watcher.snapshot()['stopped'])

    def test_registration_failure_is_recorded_and_stop_remains_safe(self):
        backend = FakeBackend({'register': OSError('unavailable')})
        watcher = runtime.PowerEventWatcher(backend)
        with self.assertRaisesRegex(OSError, 'unavailable'):
            watcher.start()
        watcher.stop()
        self.assertFalse(watcher.snapshot()['registered'])
        self.assertEqual(watcher.snapshot()['lifecycle_events'][-1]['event'], 'registration_failed')
        self.assertEqual(watcher.events, [])

    def test_failed_unregister_retains_callback_and_can_retry_but_records_failure(self):
        backend = FakeBackend({'unregister': OSError('unregister failure')})
        watcher = runtime.PowerEventWatcher(backend).start()
        with self.assertRaisesRegex(RuntimeError, 'cleanup failed'):
            watcher.stop()
        self.assertTrue(watcher.snapshot()['subscription_active'])
        self.assertFalse(watcher.snapshot()['stopped'])
        backend.emit(4)
        self.assertEqual(watcher.events[0]['code'], 4)
        backend.failures.clear()
        watcher.stop()
        self.assertTrue(watcher.snapshot()['stopped'])
        self.assertTrue(watcher.snapshot()['cleanup_errors'])

    def test_callback_error_is_a_visible_stop_signal_and_never_escapes(self):
        backend = FakeBackend()
        with runtime.PowerEventWatcher(backend) as watcher:
            with patch.object(runtime, '_event', side_effect=RuntimeError('clock failure')):
                self.assertEqual(backend.emit(4), 1)
            self.assertEqual(watcher.events[0]['event'], 'callback_error')

    def test_released_before_start_objects_cannot_begin_a_new_lifetime(self):
        request = runtime.TaskPowerRequest(FakeBackend())
        request.release()
        with self.assertRaises(RuntimeError):
            request.acquire()
        watcher = runtime.PowerEventWatcher(FakeBackend())
        watcher.stop()
        with self.assertRaises(RuntimeError):
            watcher.start()

    def test_registration_audit_failure_unregisters_before_propagating(self):
        backend = FakeBackend()
        watcher = runtime.PowerEventWatcher(backend)
        original = runtime._event
        def event(name, **details):
            if name in ('registered', 'registration_failed'):
                raise RuntimeError('audit failure')
            return original(name, **details)
        with patch.object(runtime, '_event', side_effect=event):
            with self.assertRaisesRegex(RuntimeError, 'event recording failed'):
                watcher.start()
        self.assertEqual(backend.calls, [('register',), ('unregister', 202)])
        self.assertIsNone(backend.callback)
        self.assertTrue(watcher.snapshot()['stopped'])
        self.assertTrue(watcher.snapshot()['cleanup_errors'])

    def test_successful_unregister_with_audit_failure_remains_stopped_and_fails_acceptance(self):
        backend = FakeBackend()
        watcher = runtime.PowerEventWatcher(backend).start()
        with patch.object(runtime, '_event', side_effect=OSError('clock unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'cleanup failed'):
                watcher.stop()
        self.assertIsNone(backend.callback)
        self.assertTrue(watcher.snapshot()['stopped'])
        self.assertFalse(watcher.snapshot()['subscription_active'])
        self.assertTrue(watcher.snapshot()['cleanup_errors'])

    def test_unregister_and_audit_failure_preserve_live_callback_for_retry(self):
        backend = FakeBackend({'unregister': OSError('native unregister failure')})
        watcher = runtime.PowerEventWatcher(backend).start()
        with patch.object(runtime, '_event', side_effect=RuntimeError('audit failure')):
            with self.assertRaisesRegex(RuntimeError, 'cleanup failed'):
                watcher.stop()
        self.assertTrue(watcher.snapshot()['subscription_active'])
        backend.emit(18)
        self.assertEqual(watcher.events[0]['code'], 18)
        backend.failures.clear()
        watcher.stop()
        self.assertFalse(watcher.snapshot()['subscription_active'])
        self.assertTrue(watcher.snapshot()['cleanup_errors'])


if __name__ == '__main__':
    unittest.main()
