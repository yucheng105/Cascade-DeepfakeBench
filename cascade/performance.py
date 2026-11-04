"""Bounded counters and explicitly scoped measurements, without optional packages."""
from __future__ import annotations

import math
import os
from pathlib import Path


def counter() -> dict:
    return dict(count=0, seconds_total=0., seconds_max=0.)


def observe(stats: dict, seconds: float) -> None:
    stats['count'] += 1
    stats['seconds_total'] += seconds
    stats['seconds_max'] = max(stats['seconds_max'], seconds)


def rss() -> dict:
    """Sample the current process only; OS peak is distinct from sampled RSS."""
    try:
        if os.name == 'nt':
            import ctypes
            from ctypes import wintypes
            class Counters(ctypes.Structure):
                _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
                    (name, ctypes.c_size_t) for name in ('peak', 'working', 'paged_peak',
                    'paged', 'nonpaged_peak', 'nonpaged', 'pagefile', 'pagefile_peak')]
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            psapi = ctypes.WinDLL('psapi', use_last_error=True)
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            values = Counters()
            values.cb = ctypes.sizeof(values)
            if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(values), values.cb):
                raise OSError(ctypes.get_last_error())
            return dict(bytes=values.working, os_peak_bytes=values.peak, scope='current_process_working_set')
        pages = int(Path('/proc/self/statm').read_text().split()[1])
        return dict(bytes=pages*getattr(os, 'sysconf')('SC_PAGE_SIZE'), os_peak_bytes=None, scope='current_process_resident_pages')
    except (OSError, ValueError, AttributeError) as error:
        return dict(bytes=None, os_peak_bytes=None, scope='unavailable', reason=str(error))


def distribution(values: list) -> dict:
    ordered = sorted(v for v in values if isinstance(v, (int, float)) and math.isfinite(v) and v >= 0)
    return dict(count=len(ordered), p50_seconds=ordered[math.ceil(.5*len(ordered))-1] if ordered else None,
                p95_seconds=ordered[math.ceil(.95*len(ordered))-1] if ordered else None,
                max_seconds=ordered[-1] if ordered else None)


def summarize(report: dict) -> None:
    """Exact, record-budget-bounded percentiles; do not mix latency scopes."""
    perf = report.get('performance')
    if perf is None:
        return
    scopes: dict[str, list] = {}
    for row in report['windows']:
        if row['outcome'] == 'decided' and 'latency_seconds' in row:
            scopes.setdefault(row.get('latency_scope', 'legacy_unspecified'), []).append(row['latency_seconds'])
    perf['latency_by_scope'] = {scope: distribution(values) for scope, values in scopes.items()}
    perf['parent_rss'] = rss()
    current = perf['parent_rss']['bytes']
    if current is not None:
        perf['parent_rss_sampled_peak_bytes'] = max(perf.get('parent_rss_sampled_peak_bytes', 0), current)
    perf['metadata_records'] = len(report['windows'])


def window_timing(result: dict, queue_seconds: float, roundtrip: float) -> dict:
    stages = [s['timing'] for s in result.get('stages', []) if s.get('reached')]
    profiles = [p['seconds'] for p in result.get('profiles', {}).values() if 'seconds' in p]
    def stage_sum(key):
        return sum(s[key] for s in stages) if stages and all(key in s for s in stages) else None
    return dict(queue_seconds=queue_seconds, worker_roundtrip_seconds=roundtrip,
                face_seconds=sum(profiles) if profiles else None,
                tensor_seconds=stage_sum('tensor_seconds'), forward_seconds=stage_sum('forward_seconds'),
                persistence_seconds=None,
                unavailable='null means no measurement or stage not reached; never zero-filled',
                scope='parent monotonic queue/RPC; worker local durations; face includes crop/PNG/cache; tensor includes transfer/sync; overlapping scopes are not additive')
