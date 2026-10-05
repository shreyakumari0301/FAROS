import shutil
import json
from pathlib import Path
import subprocess
import torch
print('torch:', torch.__version__, flush=True)
print('cuda:', torch.cuda.is_available(), flush=True)
print('threads:', torch.get_num_threads(), flush=True)
print('tmux:', shutil.which('tmux'), flush=True)
root = Path(__file__).resolve().parents[1]
status = root / 'outputs/phase2/status.json'
if status.exists():
    state = json.loads(status.read_text())
    print('pipeline:', state, flush=True)
    proc = Path('/proc') / str(state['pid']) / 'cmdline'
    print('live process:', proc.read_bytes().replace(b'\x00', b' ').decode() if proc.exists() else 'not running', flush=True)
    print('tmux session exists:', subprocess.run(['tmux', 'has-session', '-t', 'faros_phase2']).returncode == 0, flush=True)
