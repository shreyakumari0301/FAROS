"""Detached, resumable generation -> eight ranking probes pipeline."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--launch', action='store_true')
    p.add_argument('--threads', type=int, default=2)
    a = p.parse_args()
    os.chdir(ROOT)
    out = ROOT/'outputs/phase2'
    out.mkdir(parents=True, exist_ok=True)
    if a.launch:
        import shlex
        command = '%s -u %s --threads %d >> %s 2>&1' % (
            shlex.quote(sys.executable), shlex.quote(str(Path(__file__).resolve())),
            a.threads, shlex.quote(str(out/'pipeline.log')))
        subprocess.run(['tmux', 'new-session', '-d', '-s', 'faros_phase2', command], check=True)
        print('Started tmux session faros_phase2; log:', out/'pipeline.log')
        return
    import fcntl
    import torch
    from scripts.generate_probe_bank import generate, atomic_json
    from scripts.train_ranking_probes import run_all
    lock = open(out/'pipeline.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(a.threads)
    bank = ROOT/'data/processed/probe_bank_v2'
    started = time.time()
    def status(stage, **extra):
        atomic_json(out/'status.json', dict(stage=stage, pid=os.getpid(),
                    started_unix=started, updated_unix=time.time(), **extra))
        print('PHASE2:', stage, time.ctime(), flush=True)
    try:
        atomic_json(out/'splits.json', dict(train=list(range(80)),
                    validation=list(range(80, 90)), test=list(range(90, 100))))
        status('generating', bank=str(bank), instances=100)
        generate(bank)
        status('training', probes=8)
        run_all(bank, out)
        status('complete', comparison=str(out/'comparison.md'))
    except Exception:
        status('failed', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
