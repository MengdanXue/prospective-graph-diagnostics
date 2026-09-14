"""Task-scoped Windows power controls, imported without changing OS state.

The coordinator must acquire/start before dispatch and release/stop in finally.
Backends either succeed or raise; test backends never load a Windows library.
Power requests cannot prevent explicit sleep, so suspend/resume notifications
are an independent stop-and-review signal, including events shorter than a poll.

Native API contracts:
https://learn.microsoft.com/windows/win32/api/winbase/nf-winbase-powercreaterequest
https://learn.microsoft.com/windows/win32/api/minwinbase/ns-minwinbase-reason_context
https://learn.microsoft.com/windows/win32/api/powerbase/nf-powerbase-powerregistersuspendresumenotification
"""
from __future__ import annotations

import copy
import os
import threading
import time


SYSTEM_REQUIRED = 1
EXECUTION_REQUIRED = 3
REQUEST_TYPES = (SYSTEM_REQUIRED, EXECUTION_REQUIRED)
POWER_EVENTS = {4: 'suspend', 7: 'resume_suspend', 18: 'resume_automatic'}
DEFAULT_REASON = 'Prospective Graph: approved input robustness resource acceptance'


def _event(name, **details):
    return {'event': name, 'utc_seconds': time.time(),
            'monotonic_seconds': time.monotonic(), **details}


def _record_event(events, errors, name, *, fatal=False, **details):
    """A failed timestamp must not bypass native cleanup or erase the failure."""
    try:
        events.append(_event(name, **details))
    except BaseException as error:
        failure = {'operation': 'record_event', 'event': name, 'error': repr(error)}
        errors.append(failure)
        events.append({'event': name, **details, 'event_recording_error': repr(error)})
        if fatal:
            raise RuntimeError('power lifecycle event recording failed: ' + repr(error)) from error


def _valid_handle(handle):
    # The native backend returns integer HANDLEs; injected adapters may too.
    import ctypes
    value = getattr(handle, 'value', handle)
    return value not in (None, 0, -1, ctypes.c_void_p(-1).value)


class TaskPowerRequest:
    """One task lifetime; every successful Set is cleared and handle closed.

    ``acquired`` records that both requests once succeeded. ``released`` records
    that no request handle remains. A cleanup error still fails acceptance even
    if CloseHandle subsequently released the object. Repeated release is safe;
    acquiring this object again after release/failure is forbidden.
    """

    def __init__(self, backend=None, reason=DEFAULT_REASON):
        self._backend = backend
        self.reason = reason
        self._handle = None
        self._attempted = False
        self._active = []
        self.acquired = False
        self.released = False
        self._events = []
        self._cleanup_errors = []

    def acquire(self):
        if self._attempted or self.released:
            if (self.acquired and self._handle is not None and not self.released
                    and self._active == list(REQUEST_TYPES)):
                return self
            raise RuntimeError('power request object cannot be reused after its lifetime')
        self._attempted = True
        try:
            self._record('acquire_started', fatal=True, request_types=list(REQUEST_TYPES))
            if self._backend is None:
                self._backend = WindowsPowerBackend()
            self._handle = self._backend.create_request(self.reason)
            if not _valid_handle(self._handle):
                self._handle = None
                raise RuntimeError('power request backend returned an invalid handle')
            self._record('handle_created', fatal=True)
            for request_type in REQUEST_TYPES:
                self._backend.set_request(self._handle, request_type)
                self._active.append(request_type)
                self._record('request_set', fatal=True, request_type=request_type)
            self.acquired = True
            self._record('acquired', fatal=True)
            return self
        except BaseException as error:
            try:
                self._record('acquire_failed', error=repr(error))
            finally:
                try:
                    self.release()
                except BaseException as cleanup_error:
                    if hasattr(error, 'add_note'):
                        error.add_note('Power cleanup also failed: ' + repr(cleanup_error))
            raise

    def _record(self, name, **details):
        _record_event(self._events, self._cleanup_errors, name, **details)

    def release(self):
        errors_before = len(self._cleanup_errors)
        errors = []
        if self._handle is not None:
            try:
                for request_type in tuple(reversed(self._active)):
                    try:
                        self._backend.clear_request(self._handle, request_type)
                        self._active.remove(request_type)
                        self._record('request_cleared', request_type=request_type)
                    except BaseException as error:
                        detail = {'operation': 'clear_request', 'request_type': request_type,
                                  'error': repr(error)}
                        errors.append(detail)
                        self._record('cleanup_failed', **detail)
            finally:
                # Always close, even if a request clear or its audit record failed.
                # Retain a failed-close handle so an enclosing finally can retry.
                try:
                    self._backend.close_handle(self._handle)
                    self._handle = None
                    self._active.clear()
                    self._record('handle_closed')
                except BaseException as error:
                    detail = {'operation': 'close_handle', 'error': repr(error)}
                    errors.append(detail)
                    self._record('cleanup_failed', **detail)
        self.released = self._handle is None
        self._cleanup_errors.extend(errors)
        if len(self._cleanup_errors) > errors_before:
            raise RuntimeError('task power request cleanup failed: '
                               + repr(self._cleanup_errors[errors_before:]))
        return self.snapshot()

    def snapshot(self):
        return copy.deepcopy({'acquired': self.acquired, 'released': self.released,
                              'handle_closed': self._handle is None,
                              'active_request_types': self._active,
                              'requested_types': list(REQUEST_TYPES),
                              'reason': self.reason, 'events': self._events,
                              'cleanup_errors': self._cleanup_errors,
                              'permanent_policy_changed': False})

    def __enter__(self):
        return self.acquire()

    def __exit__(self, error_type, error, traceback):
        self.release()
        return False


class PowerEventWatcher:
    """Direct notifications only; lifecycle entries never count as power events."""

    def __init__(self, backend=None):
        self._backend = backend
        self._registration = None
        self._attempted = False
        self._registered = False
        self._stopped = False
        self._lock = threading.Lock()
        self._events = []
        self._lifecycle_events = []
        self._cleanup_errors = []

    def _callback(self, code):
        # Never throw an exception across the Windows callback boundary.
        try:
            if code in POWER_EVENTS:
                entry = _event(POWER_EVENTS[code], code=code)
                with self._lock:
                    self._events.append(entry)
            return 0
        except BaseException as error:
            with self._lock:
                self._events.append({'event': 'callback_error', 'code': None,
                                     'error': repr(error)})
            return 1

    def start(self):
        if self._attempted or self._stopped:
            if self._registration is not None and not self._stopped:
                return self
            raise RuntimeError('power event watcher cannot be reused after its lifetime')
        self._attempted = True
        try:
            self._record('registration_started', fatal=True)
            if self._backend is None:
                self._backend = WindowsPowerBackend()
            self._registration = self._backend.register_notification(self._callback)
            if not _valid_handle(self._registration):
                self._registration = None
                raise RuntimeError('power notification backend returned an invalid handle')
            self._registered = True
            self._record('registered', fatal=True)
            return self
        except BaseException as error:
            try:
                self._record('registration_failed', error=repr(error))
            finally:
                try:
                    self.stop()
                except BaseException as cleanup_error:
                    if hasattr(error, 'add_note'):
                        error.add_note('Power notification cleanup also failed: ' + repr(cleanup_error))
            raise

    def _record(self, name, **details):
        _record_event(self._lifecycle_events, self._cleanup_errors, name, **details)

    def stop(self):
        errors_before = len(self._cleanup_errors)
        if self._registration is not None:
            try:
                self._backend.unregister_notification(self._registration)
            except BaseException as error:
                detail = {'operation': 'unregister_notification', 'error': repr(error)}
                self._cleanup_errors.append(detail)
                self._record('cleanup_failed', **detail)
                raise RuntimeError('power event watcher cleanup failed: ' + repr(error)) from error
            self._registration = None
            self._record('unregistered')
        self._stopped = True
        if len(self._cleanup_errors) > errors_before:
            raise RuntimeError('power event watcher cleanup failed: '
                               + repr(self._cleanup_errors[errors_before:]))
        return self.snapshot()

    @property
    def events(self):
        with self._lock:
            return copy.deepcopy(self._events)

    def snapshot(self):
        return {'registered': self._registered, 'stopped': self._stopped,
                'subscription_active': self._registration is not None,
                'events': self.events,
                'lifecycle_events': copy.deepcopy(self._lifecycle_events),
                'cleanup_errors': copy.deepcopy(self._cleanup_errors)}

    def __enter__(self):
        return self.start()

    def __exit__(self, error_type, error, traceback):
        self.stop()
        return False


class WindowsPowerBackend:
    """Win32 adapter. Construct only in the explicitly authorized runtime.

    No display request, away mode, powercfg call, or permanent setting is used.
    Native callbacks and their structure remain alive until unregistration.
    """

    def __init__(self):
        if os.name != 'nt':
            raise RuntimeError('task-scoped power controls require Windows')
        import ctypes
        from ctypes import wintypes

        self._ctypes = ctypes
        self._kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self._power = ctypes.WinDLL('powrprof', use_last_error=True)

        class DetailedReason(ctypes.Structure):
            _fields_ = [('LocalizedReasonModule', wintypes.HMODULE),
                        ('LocalizedReasonId', wintypes.ULONG),
                        ('ReasonStringCount', wintypes.ULONG),
                        ('ReasonStrings', ctypes.POINTER(wintypes.LPWSTR))]

        class ReasonUnion(ctypes.Union):
            _fields_ = [('Detailed', DetailedReason),
                        ('SimpleReasonString', wintypes.LPWSTR)]

        class ReasonContext(ctypes.Structure):
            _fields_ = [('Version', wintypes.ULONG), ('Flags', wintypes.DWORD),
                        ('Reason', ReasonUnion)]

        callback_type = ctypes.WINFUNCTYPE(wintypes.ULONG, ctypes.c_void_p,
                                           wintypes.ULONG, ctypes.c_void_p)

        class SubscribeParameters(ctypes.Structure):
            _fields_ = [('Callback', callback_type), ('Context', ctypes.c_void_p)]

        self._reason_type = ReasonContext
        self._callback_type = callback_type
        self._subscribe_type = SubscribeParameters
        self._subscriptions = {}
        self._kernel.PowerCreateRequest.argtypes = [ctypes.POINTER(ReasonContext)]
        self._kernel.PowerCreateRequest.restype = wintypes.HANDLE
        for operation in ('PowerSetRequest', 'PowerClearRequest'):
            function = getattr(self._kernel, operation)
            function.argtypes = [wintypes.HANDLE, ctypes.c_int]
            function.restype = wintypes.BOOL
        self._kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self._kernel.CloseHandle.restype = wintypes.BOOL
        self._power.PowerRegisterSuspendResumeNotification.argtypes = [
            wintypes.DWORD, wintypes.HANDLE, ctypes.POINTER(wintypes.HANDLE)]
        self._power.PowerRegisterSuspendResumeNotification.restype = wintypes.DWORD
        self._power.PowerUnregisterSuspendResumeNotification.argtypes = [wintypes.HANDLE]
        self._power.PowerUnregisterSuspendResumeNotification.restype = wintypes.DWORD

    def _check_bool(self, value, operation):
        if not value:
            error = self._ctypes.get_last_error()
            raise OSError(error, operation + ': ' + self._ctypes.FormatError(error))

    def create_request(self, reason):
        context = self._reason_type()
        context.Version = 0  # POWER_REQUEST_CONTEXT_VERSION
        context.Flags = 1  # POWER_REQUEST_CONTEXT_SIMPLE_STRING
        context.Reason.SimpleReasonString = reason
        handle = self._kernel.PowerCreateRequest(self._ctypes.byref(context))
        if not _valid_handle(handle):
            self._check_bool(False, 'PowerCreateRequest')
        return handle

    def set_request(self, handle, request_type):
        if request_type not in REQUEST_TYPES:
            raise ValueError('only SystemRequired and ExecutionRequired are allowed')
        self._check_bool(self._kernel.PowerSetRequest(handle, request_type), 'PowerSetRequest')

    def clear_request(self, handle, request_type):
        if request_type not in REQUEST_TYPES:
            raise ValueError('unexpected power request type')
        self._check_bool(self._kernel.PowerClearRequest(handle, request_type), 'PowerClearRequest')

    def close_handle(self, handle):
        self._check_bool(self._kernel.CloseHandle(handle), 'CloseHandle')

    def register_notification(self, callback):
        native_callback = self._callback_type(lambda context, code, setting: callback(int(code)))
        parameters = self._subscribe_type(native_callback, None)
        handle = self._ctypes.c_void_p()
        error = self._power.PowerRegisterSuspendResumeNotification(
            2, self._ctypes.cast(self._ctypes.byref(parameters), self._ctypes.c_void_p),
            self._ctypes.byref(handle))  # DEVICE_NOTIFY_CALLBACK = 2
        if error:
            raise OSError(error, 'PowerRegisterSuspendResumeNotification: '
                          + self._ctypes.FormatError(error))
        if not _valid_handle(handle):
            raise RuntimeError('PowerRegisterSuspendResumeNotification returned an invalid handle')
        self._subscriptions[handle.value] = (native_callback, parameters)
        return handle.value

    def unregister_notification(self, handle):
        error = self._power.PowerUnregisterSuspendResumeNotification(handle)
        if error:
            # Keep the callback alive if unregistration failed and Windows may call it.
            raise OSError(error, 'PowerUnregisterSuspendResumeNotification: '
                          + self._ctypes.FormatError(error))
        self._subscriptions.pop(handle, None)
