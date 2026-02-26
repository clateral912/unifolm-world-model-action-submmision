"""WMA Inference Client -- submits requests to the daemon via SHM."""

import argparse
import os
import signal
import subprocess
import sys
import threading
import time

from multiprocessing import shared_memory
from multiprocessing.resource_tracker import unregister as _rt_unregister

from wma_ipc import (
    SHM_NAME, SHM_SIZE, PID_FILE, LOG_FILE,
    STATE_IDLE, STATE_REQUEST, STATE_DONE, STATE_ERROR,
    read_ctrl, write_ctrl, write_state, read_state,
    is_daemon_alive, cleanup_shm,
)

DAEMON_STARTUP_TIMEOUT = 1200  # 20 minutes


def get_parser():
    """Identical argparse to original -- user CLI unchanged."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedir",
                        type=str,
                        default=None,
                        help="Path to save the results.")
    parser.add_argument("--ckpt_path",
                        type=str,
                        default=None,
                        help="Path to the model checkpoint.")
    parser.add_argument("--config",
                        type=str,
                        help="Path to the model checkpoint.")
    parser.add_argument(
        "--prompt_dir",
        type=str,
        default=None,
        help="Directory containing videos and corresponding prompts.")
    parser.add_argument("--dataset",
                        type=str,
                        default=None,
                        help="the name of dataset to test")
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="Number of DDIM steps. If non-positive, DDPM is used instead.")
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=1.0,
        help="Eta for DDIM sampling. Set to 0.0 for deterministic results.")
    parser.add_argument("--bs",
                        type=int,
                        default=1,
                        help="Batch size for inference. Must be 1.")
    parser.add_argument("--height",
                        type=int,
                        default=320,
                        help="Height of the generated images in pixels.")
    parser.add_argument("--width",
                        type=int,
                        default=512,
                        help="Width of the generated images in pixels.")
    parser.add_argument(
        "--frame_stride",
        type=int,
        nargs='+',
        required=False,
        default=[6],
        help="frame stride control for 256 model (larger->larger motion), "
             "FPS control for 512 or 1024 model (smaller->larger motion)")
    parser.add_argument(
        "--unconditional_guidance_scale",
        type=float,
        default=1.0,
        help="Scale for classifier-free guidance during sampling.")
    parser.add_argument("--seed",
                        type=int,
                        default=123,
                        help="Random seed for reproducibility.")
    parser.add_argument("--video_length",
                        type=int,
                        default=16,
                        help="Number of frames in the generated video.")
    parser.add_argument("--num_generation",
                        type=int,
                        default=1,
                        help="seed for seed_everything")
    parser.add_argument(
        "--timestep_spacing",
        type=str,
        default="uniform",
        help="Strategy for timestep scaling.")
    parser.add_argument(
        "--guidance_rescale",
        type=float,
        default=0.0,
        help="Rescale factor for guidance.")
    parser.add_argument(
        "--perframe_ae",
        action='store_true',
        default=False,
        help="Use per-frame autoencoder decoding to reduce GPU memory usage.")
    parser.add_argument(
        "--n_action_steps",
        type=int,
        default=16,
        help="num of samples per prompt")
    parser.add_argument(
        "--exe_steps",
        type=int,
        default=16,
        help="num of samples to execute")
    parser.add_argument(
        "--n_iter",
        type=int,
        default=40,
        help="num of iteration to interact with the world model")
    parser.add_argument("--zero_pred_state",
                        action='store_true',
                        default=False,
                        help="not using the predicted states as comparison")
    parser.add_argument("--save_fps",
                        type=int,
                        default=8,
                        help="fps for the saving video")
    # New: daemon control
    parser.add_argument("--stop",
                        action="store_true",
                        default=False,
                        help="Stop the running daemon and exit.")
    parser.add_argument("--restart",
                        action="store_true",
                        default=False,
                        help="Stop existing daemon (if any), then launch a new one and submit request.")
    return parser


def stop_daemon():
    """Send SIGTERM to daemon, cleanup PID and SHM."""
    pid = is_daemon_alive()
    if pid is None:
        print("Daemon is not running.")
        return
    print(f"Stopping daemon (PID={pid})...")
    os.kill(pid, signal.SIGTERM)
    # Wait for process to exit
    for _ in range(50):  # 5 seconds max
        try:
            os.kill(pid, 0)
            time.sleep(0.1)
        except ProcessLookupError:
            break
    try:
        os.unlink(PID_FILE)
    except FileNotFoundError:
        pass
    cleanup_shm()
    print("Daemon stopped.")


def ensure_daemon(args):
    """Ensure daemon is running. Launch if not. Returns when SHM is IDLE."""
    pid = is_daemon_alive()
    if pid is not None:
        print(f"Daemon already running (PID={pid}).")
        return

    print("Daemon not running, launching...")

    # Build daemon startup args (only the model-init subset)
    daemon_cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "wma_daemon.py"),
        "--ckpt_path", args.ckpt_path,
        "--config", args.config,
        "--height", str(args.height),
        "--width", str(args.width),
        "--bs", str(args.bs),
        "--video_length", str(args.video_length),
    ]
    if args.perframe_ae:
        daemon_cmd.append("--perframe_ae")

    # Launch daemon as detached background process
    subprocess.Popen(
        daemon_cmd,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for SHM to appear and state==IDLE
    print(f"Waiting for daemon to initialize (timeout={DAEMON_STARTUP_TIMEOUT}s)...")
    t0 = time.time()
    log_pos = 0  # Track read position to avoid duplicate output
    while time.time() - t0 < DAEMON_STARTUP_TIMEOUT:
        try:
            shm = shared_memory.SharedMemory(name=SHM_NAME)
            state = read_state(shm)
            _rt_unregister(f"/{SHM_NAME}", "shared_memory")
            shm.close()
            if state == STATE_IDLE:
                print("Daemon is ready.")
                return
        except FileNotFoundError:
            pass
        # Tail daemon log while waiting — only show NEW lines
        if os.path.exists(LOG_FILE):
            try:
                with open(LOG_FILE, "r") as f:
                    f.seek(log_pos)
                    new_data = f.read()
                    log_pos = f.tell()
                    if new_data:
                        for line in new_data.splitlines():
                            if line.strip():
                                print(f"  [daemon] {line.strip()}")
            except Exception:
                pass
        time.sleep(2.0)

    raise TimeoutError(f"Daemon failed to start within {DAEMON_STARTUP_TIMEOUT}s. "
                       f"Check {LOG_FILE} for details.")


def tail_log(stop_event: threading.Event):
    """Background thread: tail -f the daemon log file."""
    while not stop_event.is_set() and not os.path.exists(LOG_FILE):
        time.sleep(0.1)
    if stop_event.is_set():
        return

    with open(LOG_FILE, "r") as f:
        # Seek to end -- only show new output
        f.seek(0, 2)
        while not stop_event.is_set():
            line = f.readline()
            if line:
                print(line, end="", flush=True)
            else:
                time.sleep(0.1)


def submit_request(args):
    """Write params to SHM control block and wait for completion."""
    shm = shared_memory.SharedMemory(name=SHM_NAME)
    try:
        # Write all params
        write_ctrl(
            shm,
            state=STATE_IDLE,
            client_pid=os.getpid(),
            exit_code=0,
            height=args.height,
            width=args.width,
            bs=args.bs,
            video_length=args.video_length,
            ddim_steps=args.ddim_steps,
            ddim_eta=args.ddim_eta,
            uncond_scale=args.unconditional_guidance_scale,
            guidance_rescale=args.guidance_rescale,
            frame_stride=args.frame_stride[0],
            n_iter=args.n_iter,
            exe_steps=args.exe_steps,
            n_action_steps=args.n_action_steps,
            seed=args.seed,
            save_fps=args.save_fps,
            perframe_ae=1 if args.perframe_ae else 0,
            zero_pred_state=1 if args.zero_pred_state else 0,
            config_path=os.path.abspath(args.config),
            ckpt_path=os.path.abspath(args.ckpt_path),
            savedir=args.savedir,
            prompt_dir=args.prompt_dir,
            dataset=args.dataset,
            timestep_spacing=args.timestep_spacing,
            error_msg="",
        )

        # Signal request
        write_state(shm, STATE_REQUEST)
        print("Request submitted, waiting for daemon...")

        # Start log tail thread
        stop_event = threading.Event()
        tail_thread = threading.Thread(target=tail_log, args=(stop_event,),
                                       daemon=True)
        tail_thread.start()

        # Poll for completion
        while True:
            state = read_state(shm)
            if state == STATE_DONE:
                stop_event.set()
                tail_thread.join(timeout=1)
                result = read_ctrl(shm)
                write_state(shm, STATE_IDLE)
                print(f"\nInference completed (exit_code={result['exit_code']}).")
                return 0
            elif state == STATE_ERROR:
                stop_event.set()
                tail_thread.join(timeout=1)
                result = read_ctrl(shm)
                write_state(shm, STATE_IDLE)
                print(f"\nInference FAILED: {result['error_msg']}",
                      file=sys.stderr)
                return 1
            time.sleep(0.1)
    finally:
        # Unregister from resource_tracker before closing — the daemon owns
        # this SHM, so the client must NOT trigger cleanup on exit.
        _rt_unregister(f"/{SHM_NAME}", "shared_memory")
        shm.close()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    if args.stop:
        stop_daemon()
        sys.exit(0)

    if args.restart:
        if is_daemon_alive():
            stop_daemon()

    ensure_daemon(args)
    rc = submit_request(args)
    sys.exit(rc)
