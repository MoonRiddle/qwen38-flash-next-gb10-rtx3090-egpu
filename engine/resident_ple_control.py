"""Read-only PLE diagnostics and guarded mmap advice and prewarm controls."""
from __future__ import annotations

import mmap
import os
import sys
import ctypes
import time
from typing import Any

_PREWARM_RESERVE_BYTES = 10 << 30
_PREWARM_BLOCK_BYTES = 4 << 20
_PLE_SCOPE = (
    "read-only mincore and proc snapshots of existing file-backed PLE mappings; "
    "page-cache residency is not RAM pinning"
)


def _page_size() -> int:
    try:
        size = int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError) as error:
        raise RuntimeError(f"system page size is unavailable: {error}") from error
    if size <= 0:
        raise RuntimeError("system page size is invalid")
    return size


def _mapping_records(tables: list[Any], page_size: int) -> list[dict[str, Any]]:
    records = []
    for table_index, table in enumerate(tables):
        for mapping_index, mapping in enumerate(_mappings(table)):
            path_value = getattr(mapping, "filename", None)
            if not isinstance(path_value, (str, bytes, os.PathLike)):
                raise RuntimeError("a PLE mapping has no file-backed filename")
            path = os.fsdecode(os.fspath(path_value))
            try:
                offset = int(mapping.offset)
                payload_bytes = int(mapping.nbytes)
                data = mapping.__array_interface__["data"]
                data_address = int(data[0])
                native_mapping = mapping._mmap
                mapped_bytes = len(native_mapping)
                file_info = os.stat(path)
            except (AttributeError, KeyError, OSError, TypeError, ValueError) as error:
                raise RuntimeError(f"PLE mapping metadata is unavailable: {error}") from error
            if offset < 0 or payload_bytes <= 0 or data_address <= 0 or mapped_bytes <= 0:
                raise RuntimeError("PLE mapping metadata contains an invalid extent")
            payload_end = offset + payload_bytes
            if payload_end > file_info.st_size:
                raise RuntimeError("PLE mapping payload extends beyond its backing file")
            map_file_start = offset - offset % page_size
            prefix_bytes = offset - map_file_start
            if mapped_bytes < prefix_bytes + payload_bytes:
                raise RuntimeError("PLE mapping length does not cover its payload extent")
            map_address = data_address - prefix_bytes
            if map_address <= 0 or map_address % page_size:
                raise RuntimeError("PLE mapping base address is not page aligned")
            records.append(
                {
                    "table_index": table_index,
                    "mapping_index": mapping_index,
                    "table": table,
                    "mapping": mapping,
                    "path": path,
                    "device": int(file_info.st_dev),
                    "inode": int(file_info.st_ino),
                    "file_bytes": int(file_info.st_size),
                    "payload_start": offset,
                    "payload_end": payload_end,
                    "payload_bytes": payload_bytes,
                    "map_file_start": map_file_start,
                    "map_address": map_address,
                    "mapped_bytes": mapped_bytes,
                    "unique_ranges": [],
                    "unique_page_ranges": [],
                }
            )
    if not records:
        raise RuntimeError("the loaded PLE tables have no mapped payload extents")
    return records


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _subtract_ranges(
    start: int, end: int, covered: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    result = []
    cursor = start
    for covered_start, covered_end in covered:
        if covered_end <= cursor:
            continue
        if covered_start >= end:
            break
        if covered_start > cursor:
            result.append((cursor, min(covered_start, end)))
        cursor = max(cursor, covered_end)
        if cursor >= end:
            break
    if cursor < end:
        result.append((cursor, end))
    return result


def _prepare_unique_ranges(records: list[dict[str, Any]], page_size: int) -> None:
    covered_bytes: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for record in records:
        file_key = (record["device"], record["inode"])
        prior = covered_bytes.get(file_key, [])
        unique = _subtract_ranges(
            record["payload_start"], record["payload_end"], prior
        )
        record["unique_ranges"] = unique
        covered_bytes[file_key] = _merge_ranges(prior + unique)

    covered_pages: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for record in records:
        file_key = (record["device"], record["inode"])
        page_ranges = _merge_ranges(
            [
                (start // page_size, (end + page_size - 1) // page_size)
                for start, end in record["unique_ranges"]
            ]
        )
        prior = covered_pages.get(file_key, [])
        unique_pages = []
        for start, end in page_ranges:
            unique_pages.extend(_subtract_ranges(start, end, prior))
        record["unique_page_ranges"] = unique_pages
        covered_pages[file_key] = _merge_ranges(prior + unique_pages)


def _mincore_pages(address: int, length: int, page_size: int) -> bytes:
    if address <= 0 or address % page_size or length <= 0 or length % page_size:
        raise RuntimeError("mincore range must be positive and page aligned")
    page_count = length // page_size
    try:
        function = ctypes.CDLL(None, use_errno=True).mincore
    except (AttributeError, OSError) as error:
        raise OSError("mincore is unavailable in the process libc") from error
    function.argtypes = (
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_ubyte),
    )
    function.restype = ctypes.c_int
    vector = (ctypes.c_ubyte * page_count)()
    result = function(
        ctypes.c_void_p(address),
        ctypes.c_size_t(length),
        vector,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    return bytes(vector)


def _measure_records(
    records: list[dict[str, Any]], page_size: int
) -> dict[str, Any]:
    _prepare_unique_ranges(records, page_size)
    residency: dict[tuple[int, int], list[tuple[int, int, bytes]]] = {}
    payload_pages = 0
    resident_pages = 0
    mincore_calls = 0
    for record in records:
        file_key = (record["device"], record["inode"])
        map_page_start = record["map_file_start"] // page_size
        map_page_end = (
            record["map_file_start"] + record["mapped_bytes"] + page_size - 1
        ) // page_size
        for page_start, page_end in record["unique_page_ranges"]:
            if page_start < map_page_start or page_end > map_page_end:
                raise RuntimeError("mincore page range exceeds its live mmap")
            address = record["map_address"] + (
                page_start * page_size - record["map_file_start"]
            )
            length = (page_end - page_start) * page_size
            vector = _mincore_pages(address, length, page_size)
            if len(vector) != page_end - page_start:
                raise RuntimeError("mincore returned an incomplete residency vector")
            residency.setdefault(file_key, []).append(
                (page_start, page_end, vector)
            )
            payload_pages += page_end - page_start
            resident_pages += sum(bool(value & 1) for value in vector)
            mincore_calls += 1

    for record in records:
        file_key = (record["device"], record["inode"])
        segments = residency.get(file_key, [])
        resident_bytes = 0
        for start, end in record["unique_ranges"]:
            first_page = start // page_size
            end_page = (end + page_size - 1) // page_size
            for segment_start, segment_end, vector in segments:
                overlap_start = max(first_page, segment_start)
                overlap_end = min(end_page, segment_end)
                for page_index in range(overlap_start, overlap_end):
                    if vector[page_index - segment_start] & 1:
                        resident_bytes += max(
                            0,
                            min(end, (page_index + 1) * page_size)
                            - max(start, page_index * page_size),
                        )
        record["resident_payload_bytes"] = resident_bytes
    return {
        "payload_intersecting_pages": payload_pages,
        "resident_pages_intersecting_payload": resident_pages,
        "mincore_calls": mincore_calls,
    }


def _proc_values(path: str) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="ascii") as stream:
            text = stream.read()
        values: dict[str, Any] = {}
        for line in text.splitlines():
            name, separator, raw = line.partition(":")
            if not separator:
                continue
            parts = raw.split()
            if not parts:
                continue
            try:
                value = int(parts[0])
            except ValueError:
                continue
            if len(parts) > 1 and parts[1] == "kB":
                values[f"{name}_bytes"] = value * 1024
            else:
                values[name] = value
        return {"available": True, "values": values, "error": None}
    except (OSError, UnicodeError) as error:
        return {
            "available": False,
            "values": None,
            "error": f"{type(error).__name__}: {error}",
        }


def _proc_stat_values(path: str) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="ascii") as stream:
            text = stream.read()
        close = text.rfind(")")
        if close < 0:
            raise ValueError("comm delimiter is missing")
        fields = text[close + 1 :].split()
        if len(fields) <= 10:
            raise ValueError("stat has too few fields after comm")
        values = {
            "minor_faults": int(fields[7]),
            "child_minor_faults": int(fields[8]),
            "major_faults": int(fields[9]),
            "child_major_faults": int(fields[10]),
        }
        return {"available": True, "values": values, "error": None}
    except (OSError, UnicodeError, ValueError) as error:
        return {
            "available": False,
            "values": None,
            "error": f"{type(error).__name__}: {error}",
        }


def _process_snapshot(proc_root: str = "/proc/self") -> dict[str, Any]:
    proc_parent = os.path.dirname(os.path.normpath(proc_root))
    return {
        "io": _proc_values(os.path.join(proc_root, "io")),
        "stat": _proc_stat_values(os.path.join(proc_root, "stat")),
        "smaps_rollup": _proc_values(os.path.join(proc_root, "smaps_rollup")),
        "status": _proc_values(os.path.join(proc_root, "status")),
        "meminfo": _proc_values(os.path.join(proc_parent, "meminfo")),
        "units": "kB memory fields are converted to bytes; other counters retain proc units",
    }


def _memavailable_bytes(proc_root: str = "/proc/self") -> int:
    proc_parent = os.path.dirname(os.path.normpath(proc_root))
    snapshot = _proc_values(os.path.join(proc_parent, "meminfo"))
    if snapshot["available"] is not True:
        raise RuntimeError(f"MemAvailable is unavailable: {snapshot['error']}")
    available = snapshot["values"].get("MemAvailable_bytes")
    if type(available) is not int or available < 0:
        raise RuntimeError("MemAvailable is missing or invalid in proc meminfo")
    return available


def _gather_counters() -> dict[str, Any]:
    module = sys.modules.get("vllm_ple_mmap")
    stats = getattr(module, "_STATS", None)
    if not isinstance(stats, dict):
        return {
            "available": False,
            "source": "vllm_ple_mmap._STATS",
            "values": None,
            "error": "the installed PLE gather counters are unavailable",
        }
    values = {}
    for key in ("calls", "op_ms", "gather_ms", "rows", "bytes", "pf_hit", "pf_miss"):
        value = stats.get(key)
        if type(value) in (int, float):
            values[key] = value
    last = getattr(module, "_STATS_LAST", None)
    window_start = None
    if isinstance(last, list) and last and type(last[0]) in (int, float):
        window_start = float(last[0])
    return {
        "available": True,
        "source": "vllm_ple_mmap._STATS",
        "values": values,
        "window_started_at_monotonic_seconds": window_start,
        "windowed_fields": ["calls", "op_ms", "gather_ms", "rows", "bytes"],
        "cumulative_fields": ["pf_hit", "pf_miss"],
        "reset_note": "_stats_log resets windowed fields; compare only within one unchanged window",
        "error": None,
    }


def _table_entries(
    tables: list[Any], records: list[dict[str, Any]], mincore_available: bool
) -> list[dict[str, Any]]:
    entries = []
    for table_index, table in enumerate(tables):
        table_records = [
            record for record in records if record["table_index"] == table_index
        ]
        entries.append(
            {
                "backend": type(table).__name__,
                "mode": _mode(table),
                "mapped_planes": len(table_records),
                "payload_bytes": sum(
                    record["payload_bytes"] for record in table_records
                ),
                "unique_payload_bytes": sum(
                    sum(end - start for start, end in record["unique_ranges"])
                    for record in table_records
                ),
                "resident_payload_bytes": (
                    sum(
                        record["resident_payload_bytes"]
                        for record in table_records
                    )
                    if mincore_available
                    else None
                ),
                "mappings": [
                    {
                        "path": record["path"],
                        "file_device": record["device"],
                        "file_inode": record["inode"],
                        "file_bytes": record["file_bytes"],
                        "payload_offset_bytes": record["payload_start"],
                        "payload_bytes": record["payload_bytes"],
                        "unique_payload_bytes": sum(
                            end - start for start, end in record["unique_ranges"]
                        ),
                        "resident_payload_bytes": (
                            record["resident_payload_bytes"]
                            if mincore_available
                            else None
                        ),
                    }
                    for record in table_records
                ],
            }
        )
    return entries


def _tables() -> list[Any]:
    module = sys.modules.get('vllm_ple_mmap')
    registry = getattr(module, '_REGISTRY', None)
    if not isinstance(registry, dict) or not registry:
        raise RuntimeError('the loaded PLE mmap registry is unavailable')
    tables: list[Any] = []
    seen: set[int] = set()
    for layer in registry.values():
        table = getattr(getattr(layer, 'ngram_embedding', None), 'table', None)
        if table is None:
            raise RuntimeError('a registered PLE embedding has no loaded mmap table')
        table = getattr(table, 'inner', table)
        if id(table) in seen:
            continue
        mappings = _mappings(table)
        if not mappings:
            raise RuntimeError('a loaded PLE table has no original mapped file planes')
        tables.append(table)
        seen.add(id(table))
    return tables


def _mappings(table: Any) -> list[Any]:
    result = []
    for field in ('mm', 'scale_mm'):
        for mapping in getattr(table, field, []):
            if mapping is not None:
                if not callable(getattr(getattr(mapping, '_mmap', None), 'madvise', None)):
                    raise RuntimeError('a PLE file plane does not support mmap.madvise')
                result.append(mapping)
    return result


def _mode(table: Any) -> str:
    value = getattr(table, '_resident_madvise_mode', getattr(table, 'madvise_mode', None))
    if value is None:
        value = 'random' if os.environ.get('VLLM_PLE_MMAP_MADV_RANDOM', '0') == '1' else 'normal'
    if value not in ('normal', 'random'):
        raise RuntimeError('the loaded PLE mapping hint is unknown')
    return value


def ple_mmap_status() -> dict[str, Any]:
    try:
        tables = _tables()
        entries = [
            {'backend':type(table).__name__, 'mode':_mode(table), 'mapped_planes':len(_mappings(table)),
             'mapped_payload_bytes':sum(int(mapping.nbytes) for mapping in _mappings(table))}
            for table in tables
        ]
        return {'ready':True, 'tables':entries, 'reason':None, 'pin_claimed':False,
                'scope':'existing file-backed mappings only; no tensor or hash changes'}
    except (RuntimeError, TypeError, AttributeError) as error:
        return {'ready':False, 'tables':[], 'reason':str(error), 'pin_claimed':False}



def ple_mmap_diagnostics() -> dict[str, Any]:
    """Take an opt-in, read-only residency and process-counter snapshot."""
    process = _process_snapshot()
    gather_counters = _gather_counters()
    page_size = None
    try:
        page_size = _page_size()
        tables = _tables()
        records = _mapping_records(tables, page_size)
    except (OSError, RuntimeError, TypeError, AttributeError, ValueError) as error:
        return {
            "schema": "ple-mmap-diagnostics-v1",
            "ready": False,
            "reason": str(error),
            "sampled_at_monotonic_ns": time.monotonic_ns(),
            "page_size": page_size,
            "mincore": {"available": False, "error": str(error)},
            "totals": {
                "payload_bytes": None,
                "unique_payload_bytes": None,
                "resident_payload_bytes": None,
                "missing_payload_bytes": None,
            },
            "tables": [],
            "process": process,
            "gather_counters": gather_counters,
            "pin_claimed": False,
            "scope": _PLE_SCOPE,
        }

    try:
        page_counts = _measure_records(records, page_size)
        mincore = {
            "available": True,
            "error": None,
            "page_size": page_size,
            **page_counts,
        }
        mincore_available = True
        resident_payload_bytes = sum(
            record["resident_payload_bytes"] for record in records
        )
    except (OSError, RuntimeError, TypeError, AttributeError, ValueError) as error:
        mincore_error = f"{type(error).__name__}: {error}"
        mincore = {
            "available": False,
            "error": mincore_error,
            "page_size": page_size,
            "payload_intersecting_pages": None,
            "resident_pages_intersecting_payload": None,
            "mincore_calls": None,
        }
        mincore_available = False
        resident_payload_bytes = None
    raw_payload_bytes = sum(record["payload_bytes"] for record in records)
    unique_payload_bytes = sum(
        end - start
        for record in records
        for start, end in record["unique_ranges"]
    )

    try:
        entries = _table_entries(tables, records, mincore_available)
    except (RuntimeError, TypeError, AttributeError, ValueError) as error:
        return {
            "schema": "ple-mmap-diagnostics-v1",
            "ready": False,
            "reason": str(error),
            "sampled_at_monotonic_ns": time.monotonic_ns(),
            "page_size": page_size,
            "mincore": mincore,
            "totals": {
                "payload_bytes": raw_payload_bytes,
                "unique_payload_bytes": unique_payload_bytes,
                "resident_payload_bytes": resident_payload_bytes,
                "missing_payload_bytes": (
                    unique_payload_bytes - resident_payload_bytes
                    if resident_payload_bytes is not None
                    else None
                ),
            },
            "tables": [],
            "process": process,
            "gather_counters": gather_counters,
            "pin_claimed": False,
            "scope": _PLE_SCOPE,
        }

    return {
        "schema": "ple-mmap-diagnostics-v1",
        "ready": True,
        "reason": None,
        "sampled_at_monotonic_ns": time.monotonic_ns(),
        "page_size": page_size,
        "mincore": mincore,
        "totals": {
            "payload_bytes": raw_payload_bytes,
            "unique_payload_bytes": unique_payload_bytes,
            "resident_payload_bytes": resident_payload_bytes,
            "missing_payload_bytes": (
                unique_payload_bytes - resident_payload_bytes
                if resident_payload_bytes is not None
                else None
            ),
        },
        "tables": entries,
        "process": process,
        "gather_counters": gather_counters,
        "pin_claimed": False,
        "scope": _PLE_SCOPE,
    }


def _unique_file_extents(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    extents = []
    for record in records:
        for start, end in record["unique_ranges"]:
            extents.append(
                {
                    "device": record["device"],
                    "inode": record["inode"],
                    "path": record["path"],
                    "start": start,
                    "end": end,
                }
            )
    extents.sort(
        key=lambda extent: (
            extent["device"],
            extent["inode"],
            extent["start"],
            extent["end"],
        )
    )
    merged: list[dict[str, Any]] = []
    for extent in extents:
        if (
            merged
            and (merged[-1]["device"], merged[-1]["inode"])
            == (extent["device"], extent["inode"])
            and extent["start"] <= merged[-1]["end"]
        ):
            merged[-1]["end"] = max(merged[-1]["end"], extent["end"])
        else:
            merged.append(dict(extent))
    return merged


def _guarded_stream(
    extents: list[dict[str, Any]],
    *,
    payload_bytes: int,
    resident_payload_bytes: int,
    initial_available_bytes: int,
    available_bytes: Any,
    opener: Any = None,
    reserve_bytes: int = _PREWARM_RESERVE_BYTES,
    block_bytes: int = _PREWARM_BLOCK_BYTES,
) -> dict[str, Any]:
    missing_bytes = payload_bytes - resident_payload_bytes
    result = {
        "schema": "ple-mmap-prewarm-v1",
        "status": "skipped",
        "reason": None,
        "payload_bytes": payload_bytes,
        "resident_payload_bytes": resident_payload_bytes,
        "missing_payload_bytes": missing_bytes,
        "mem_available_before_bytes": initial_available_bytes,
        "mem_available_after_bytes": initial_available_bytes,
        "reserve_bytes": reserve_bytes,
        "initial_budget_passed": (
            initial_available_bytes >= missing_bytes + reserve_bytes
        ),
        "reserve_checks": 0,
        "minimum_mem_available_observed_bytes": initial_available_bytes,
        "reserve_passed_at_last_check": initial_available_bytes >= reserve_bytes,
        "bytes_read": 0,
        "ranges_total": len(extents),
        "ranges_started": 0,
        "ranges_read": 0,
        "buffer_bytes": block_bytes,
        "pin_claimed": False,
    }
    if reserve_bytes < 0 or block_bytes <= 0:
        result["reason"] = "invalid prewarm reserve or block size"
        return result
    result["reserve_checks"] = 1
    if initial_available_bytes < missing_bytes + reserve_bytes:
        result["reason"] = (
            "MemAvailable is below missing PLE payload bytes plus the reserve"
        )
        return result

    if opener is None:
        opener = lambda path: open(path, "rb", buffering=0)
    buffer = bytearray(block_bytes)
    result["status"] = "completed"

    def recheck_reserve() -> None:
        available = available_bytes()
        result["reserve_checks"] += 1
        result["mem_available_after_bytes"] = available
        result["minimum_mem_available_observed_bytes"] = min(
            result["minimum_mem_available_observed_bytes"], available
        )
        result["reserve_passed_at_last_check"] = available >= reserve_bytes

    for extent in extents:
        try:
            result["ranges_started"] += 1
            with opener(extent["path"]) as stream:
                file_info = os.fstat(stream.fileno())
                if (int(file_info.st_dev), int(file_info.st_ino)) != (
                    extent["device"],
                    extent["inode"],
                ):
                    raise RuntimeError("PLE backing file identity changed before prewarm")
                if file_info.st_size < extent["end"]:
                    raise RuntimeError("PLE backing file became shorter before prewarm")
                stream.seek(extent["start"])
                remaining = extent["end"] - extent["start"]
                while remaining > 0:
                    recheck_reserve()
                    if not result["reserve_passed_at_last_check"]:
                        result["status"] = "aborted_reserve"
                        result["reason"] = (
                            "MemAvailable fell below the reserve during prewarm"
                        )
                        return result
                    requested = min(block_bytes, remaining)
                    count = stream.readinto(memoryview(buffer)[:requested])
                    if count is None or count <= 0:
                        result["status"] = "failed"
                        result["reason"] = (
                            "short read reached EOF before the mapped payload ended"
                        )
                        return result
                    if count > requested:
                        result["status"] = "failed"
                        result["reason"] = "readinto returned more bytes than requested"
                        return result
                    remaining -= count
                    result["bytes_read"] += count
                    if remaining == 0:
                        result["ranges_read"] += 1
                    recheck_reserve()
                    if not result["reserve_passed_at_last_check"]:
                        result["status"] = "aborted_reserve"
                        result["reason"] = (
                            "MemAvailable fell below the reserve during prewarm"
                        )
                        return result
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            result["status"] = "failed"
            result["reason"] = f"{type(error).__name__}: {error}"
            return result
    return result


def prewarm_ple_table(table: Any) -> dict[str, Any]:
    """Run an explicit diagnostic prewarm only when the RAM reserve is safe."""
    try:
        page_size = _page_size()
        records = _mapping_records([table], page_size)
        _measure_records(records, page_size)
        payload_bytes = sum(
            end - start
            for record in records
            for start, end in record["unique_ranges"]
        )
        resident_payload_bytes = sum(
            record["resident_payload_bytes"] for record in records
        )
    except (OSError, RuntimeError, TypeError, AttributeError, ValueError) as error:
        return {
            "schema": "ple-mmap-prewarm-v1",
            "status": "skipped",
            "reason": f"PLE residency could not be measured: {type(error).__name__}: {error}",
            "payload_bytes": None,
            "resident_payload_bytes": None,
            "missing_payload_bytes": None,
            "mem_available_before_bytes": None,
            "bytes_read": 0,
            "pin_claimed": False,
        }
    try:
        available = _memavailable_bytes()
    except RuntimeError as error:
        return {
            "schema": "ple-mmap-prewarm-v1",
            "status": "skipped",
            "reason": str(error),
            "payload_bytes": payload_bytes,
            "resident_payload_bytes": resident_payload_bytes,
            "missing_payload_bytes": payload_bytes - resident_payload_bytes,
            "mem_available_before_bytes": None,
            "bytes_read": 0,
            "pin_claimed": False,
        }
    return _guarded_stream(
        _unique_file_extents(records),
        payload_bytes=payload_bytes,
        resident_payload_bytes=resident_payload_bytes,
        initial_available_bytes=available,
        available_bytes=_memavailable_bytes,
    )

def guarded_prewarm_registered_tables() -> dict[str, Any]:
    """Stream full deduplicated payload extents when PLE bytes are missing.

    Mincore selects whether to run, not individual pages; result scope and byte
    counts distinguish this from a missing-page-only read.
    """
    before = ple_mmap_diagnostics()
    before_totals = before.get("totals")
    before_totals = before_totals if isinstance(before_totals, dict) else {}
    payload_bytes = before_totals.get("unique_payload_bytes")
    resident_before = before_totals.get("resident_payload_bytes")
    missing_before = before_totals.get("missing_payload_bytes")

    def snapshot_mem_available(snapshot: dict[str, Any]) -> int | None:
        process = snapshot.get("process")
        meminfo = process.get("meminfo") if isinstance(process, dict) else None
        values = meminfo.get("values") if isinstance(meminfo, dict) else None
        available = (
            values.get("MemAvailable_bytes") if isinstance(values, dict) else None
        )
        return available if type(available) is int and available >= 0 else None

    before_available = snapshot_mem_available(before)

    def finish(stream: dict[str, Any]) -> dict[str, Any]:
        after = ple_mmap_diagnostics()
        after_totals = after.get("totals")
        after_totals = after_totals if isinstance(after_totals, dict) else {}
        after_missing = after_totals.get("missing_payload_bytes")
        after_available = snapshot_mem_available(after)
        status = stream.get("status", "failed")
        bytes_read = stream.get("bytes_read", 0)
        full_stream_complete = (
            bytes_read == stream.get("payload_bytes", payload_bytes)
            and stream.get("ranges_read") == stream.get("ranges_total")
        )
        if status == "completed" and not full_stream_complete:
            status = "failed"
        reason = stream.get("reason")
        if status == "failed" and reason is None:
            reason = "payload stream ended without reading the full planned extent"
        return {
            "schema": "ple-mmap-registered-prewarm-v1",
            "status": status,
            "operation_complete": status in ("completed", "noop"),
            "payload_stream_complete": (
                None if status == "noop" else full_stream_complete
            ),
            "reason": reason,
            "planned_stream_scope": stream.get("planned_stream_scope", "none"),
            "stream_scope": (
                "full_unique_payload_ranges"
                if full_stream_complete
                else "partial_unique_payload_ranges"
                if bytes_read > 0
                else "none"
            ),
            "payload_bytes": stream.get("payload_bytes", payload_bytes),
            "resident_payload_bytes_before": stream.get(
                "resident_payload_bytes", resident_before
            ),
            "missing_payload_bytes_before": stream.get(
                "missing_payload_bytes", missing_before
            ),
            "resident_payload_bytes_after": after_totals.get(
                "resident_payload_bytes"
            ),
            "missing_payload_bytes_after": after_missing,
            "bytes_read": bytes_read,
            "ranges_total": stream.get("ranges_total", 0),
            "ranges_started": stream.get("ranges_started", 0),
            "ranges_read": stream.get("ranges_read", 0),
            "buffer_bytes": stream.get("buffer_bytes", _PREWARM_BLOCK_BYTES),
            "reserve_bytes": stream.get("reserve_bytes", _PREWARM_RESERVE_BYTES),
            "mem_available_before_bytes": stream.get(
                "mem_available_before_bytes", before_available
            ),
            "mem_available_after_bytes": stream.get(
                "mem_available_after_bytes", after_available
            ),
            "mem_available_after_diagnostics_bytes": after_available,
            "initial_budget_passed": stream.get("initial_budget_passed"),
            "reserve_checks": stream.get("reserve_checks", 0),
            "minimum_mem_available_observed_bytes": stream.get(
                "minimum_mem_available_observed_bytes", before_available
            ),
            "reserve_passed_at_last_check": stream.get(
                "reserve_passed_at_last_check"
            ),
            "reserve_passed_after_diagnostics": (
                after_available >= _PREWARM_RESERVE_BYTES
                if after_available is not None
                else None
            ),
            "all_payload_resident_after": (
                after_missing == 0 if type(after_missing) is int else None
            ),
            "diagnostics_before": before,
            "diagnostics_after": after,
            "pin_claimed": False,
        }

    def skipped(reason: str) -> dict[str, Any]:
        return finish(
            {
                "status": "skipped",
                "reason": reason,
                "payload_bytes": payload_bytes,
                "resident_payload_bytes": resident_before,
                "missing_payload_bytes": missing_before,
                "mem_available_before_bytes": before_available,
                "mem_available_after_bytes": before_available,
                "reserve_bytes": _PREWARM_RESERVE_BYTES,
                "initial_budget_passed": None,
                "reserve_checks": 0,
                "minimum_mem_available_observed_bytes": before_available,
                "reserve_passed_at_last_check": None,
                "bytes_read": 0,
                "ranges_total": 0,
                "ranges_started": 0,
                "ranges_read": 0,
                "buffer_bytes": _PREWARM_BLOCK_BYTES,
            }
        )

    if before.get("ready") is not True:
        return skipped(
            str(before.get("reason") or "registered PLE diagnostics are unavailable")
        )
    mincore = before.get("mincore")
    if not isinstance(mincore, dict) or mincore.get("available") is not True:
        error = mincore.get("error") if isinstance(mincore, dict) else None
        return skipped(str(error or "PLE mincore residency is unavailable"))
    if (
        type(payload_bytes) is not int
        or type(resident_before) is not int
        or type(missing_before) is not int
        or payload_bytes < 0
        or resident_before < 0
        or missing_before < 0
        or resident_before + missing_before != payload_bytes
    ):
        return skipped("PLE payload residency totals are missing or inconsistent")
    if missing_before == 0:
        return finish(
            {
                "status": "noop",
                "reason": None,
                "payload_bytes": payload_bytes,
                "resident_payload_bytes": resident_before,
                "missing_payload_bytes": 0,
                "mem_available_before_bytes": before_available,
                "mem_available_after_bytes": before_available,
                "reserve_bytes": _PREWARM_RESERVE_BYTES,
                "initial_budget_passed": None,
                "reserve_checks": 0,
                "minimum_mem_available_observed_bytes": before_available,
                "reserve_passed_at_last_check": None,
                "bytes_read": 0,
                "ranges_total": 0,
                "ranges_started": 0,
                "ranges_read": 0,
                "buffer_bytes": _PREWARM_BLOCK_BYTES,
            }
        )

    try:
        page_size = _page_size()
        records = _mapping_records(_tables(), page_size)
        _prepare_unique_ranges(records, page_size)
        extents = _unique_file_extents(records)
        extent_bytes = sum(extent["end"] - extent["start"] for extent in extents)
        if extent_bytes != payload_bytes:
            raise RuntimeError("registered PLE payload ranges changed after diagnostics")
        available = _memavailable_bytes()
    except (OSError, RuntimeError, TypeError, AttributeError, ValueError) as error:
        return skipped(f"{type(error).__name__}: {error}")

    stream = _guarded_stream(
        extents,
        payload_bytes=payload_bytes,
        resident_payload_bytes=resident_before,
        initial_available_bytes=available,
        available_bytes=_memavailable_bytes,
    )
    stream["planned_stream_scope"] = "full_unique_payload_ranges"
    return finish(stream)


def configure_ple_madvise(mode: str) -> dict[str, Any]:
    if mode not in ('normal', 'random'):
        raise ValueError('PLE mmap hint must be normal or random')
    tables = _tables()
    work = [(table, _mode(table), _mappings(table)) for table in tables]
    hint = mmap.MADV_NORMAL if mode == 'normal' else mmap.MADV_RANDOM
    touched = []
    try:
        for table, previous, mappings in work:
            for mapping in mappings:
                mapping._mmap.madvise(hint)
                touched.append((mapping, previous))
    except OSError as error:
        rollback_errors = []
        for mapping, previous in reversed(touched):
            try:
                mapping._mmap.madvise(mmap.MADV_NORMAL if previous == 'normal' else mmap.MADV_RANDOM)
            except OSError as rollback_error:
                rollback_errors.append(str(rollback_error))
        raise RuntimeError(f'PLE mmap hint application failed: {error}; rollback_errors={rollback_errors}') from error
    for table, _previous, _mappings_unused in work:
        table._resident_madvise_mode = mode
        if hasattr(table, 'madvise_mode'):
            table.madvise_mode = mode
    return ple_mmap_status()
