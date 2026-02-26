"""WMA Daemon IPC protocol -- shared memory control block layout and helpers."""

import os
import struct
import signal
from multiprocessing import shared_memory
from multiprocessing.resource_tracker import unregister as _rt_unregister

# -- Constants --
SHM_NAME = "wma_ctrl"
SHM_SIZE = 4096
PID_FILE = "/tmp/wma_daemon.pid"
LOG_FILE = "/tmp/wma_daemon.log"

# -- State Machine --
STATE_IDLE = 0
STATE_REQUEST = 1
STATE_RUNNING = 2
STATE_DONE = 3
STATE_ERROR = 4

# -- Control Block Layout --
# Numeric fields: struct format string
# I=uint32, i=int32, f=float32
_NUM_FMT = "Iii" + "iiii" + "iff" + "fiiii" + "iiii"
_NUM_SIZE = struct.calcsize(_NUM_FMT)  # 76 bytes

# String fields: (name, max_len)
_STR_FIELDS = [
    ("config_path",      256),   # offset 76
    ("ckpt_path",        256),   # offset 332
    ("savedir",          256),   # offset 588
    ("prompt_dir",       256),   # offset 844
    ("dataset",          256),   # offset 1100
    ("timestep_spacing", 128),   # offset 1356
    ("error_msg",        512),   # offset 1484
]

# Numeric field names in order (must match _NUM_FMT)
_NUM_FIELDS = [
    "state", "client_pid", "exit_code",
    "height", "width", "bs", "video_length",
    "ddim_steps", "ddim_eta", "uncond_scale",
    "guidance_rescale", "frame_stride", "n_iter", "exe_steps",
    "n_action_steps", "seed", "save_fps", "perframe_ae",
    "zero_pred_state",
]


def write_ctrl(shm: shared_memory.SharedMemory, **kwargs) -> None:
    """Write fields into the SHM control block. Only writes fields present in kwargs."""
    buf = shm.buf

    # Read current numeric values, update selectively
    nums = list(struct.unpack_from(_NUM_FMT, buf, 0))
    for i, name in enumerate(_NUM_FIELDS):
        if name in kwargs:
            nums[i] = kwargs[name]
    struct.pack_into(_NUM_FMT, buf, 0, *nums)

    # Write string fields
    offset = _NUM_SIZE
    for name, maxlen in _STR_FIELDS:
        if name in kwargs:
            val = kwargs[name] or ""
            encoded = val.encode("utf-8")[:maxlen - 1] + b"\x00"
            buf[offset:offset + len(encoded)] = encoded
            # Zero-fill remainder
            remainder = maxlen - len(encoded)
            if remainder > 0:
                buf[offset + len(encoded):offset + maxlen] = b"\x00" * remainder
        offset += maxlen


def read_ctrl(shm: shared_memory.SharedMemory) -> dict:
    """Read all fields from the SHM control block."""
    buf = bytes(shm.buf[:SHM_SIZE])

    nums = struct.unpack_from(_NUM_FMT, buf, 0)
    result = {name: val for name, val in zip(_NUM_FIELDS, nums)}

    offset = _NUM_SIZE
    for name, maxlen in _STR_FIELDS:
        raw = buf[offset:offset + maxlen]
        result[name] = raw.split(b"\x00", 1)[0].decode("utf-8")
        offset += maxlen

    return result


def read_state(shm: shared_memory.SharedMemory) -> int:
    """Read only the state field (fast path for polling)."""
    return struct.unpack_from("I", shm.buf, 0)[0]


def write_state(shm: shared_memory.SharedMemory, state: int) -> None:
    """Write only the state field."""
    struct.pack_into("I", shm.buf, 0, state)


def is_daemon_alive() -> int | None:
    """Check if daemon is running. Returns PID if alive, None otherwise."""
    if not os.path.exists(PID_FILE):
        return None
    try:
        with open(PID_FILE) as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)  # Check process exists
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        # Stale PID file -- clean up
        try:
            os.unlink(PID_FILE)
        except FileNotFoundError:
            pass
        return None


def cleanup_shm() -> None:
    """Remove the SHM block if it exists."""
    try:
        shm = shared_memory.SharedMemory(name=SHM_NAME)
        # Python 3.10 registers SHM with resource_tracker even when opening
        # existing (create=False). Unregister BEFORE close/unlink so the
        # tracker doesn't try to double-clean on process exit.
        _rt_unregister(f"/{SHM_NAME}", "shared_memory")
        shm.close()
        shm.unlink()
    except FileNotFoundError:
        pass
