"""Test frozen protocol, paired statistics, and synthetic-only end-to-end evaluation."""
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.run_phase2c as c


def main():
    torch.set_num_threads(2)
    base = dict(costs=np.array([[1., 0.]]*4), valid=np.ones((4, 2), bool),
                nonterminal=np.ones(4, bool), instance=np.array([0, 0, 1, 1]))
    strong = dict(base, scores=np.array([[0., 1.]]*4))
    bad = dict(base, scores=np.array([[1., 0.]]*4))
    h = c.hypotheses(dict(strong_weighted=strong, policy=bad, structural_weighted=bad), draws=20)
    assert all(x['delta'] == -1 and x['ci95'] == [-1., -1.] and x['supported'] for x in h.values())
    try:
        c.hypotheses(dict(strong_weighted=strong, policy=dict(bad, instance=np.array([1, 1, 0, 0])),
                          structural_weighted=bad), draws=20)
        raise AssertionError('Unpaired data accepted')
    except ValueError:
        pass
    print('PASS: hypothesis signs, cluster intervals, record alignment', flush=True)
    with tempfile.TemporaryDirectory(prefix='phase2c_test_') as temp:
        out, bank = Path(temp)/'outputs', Path(temp)/'bank'
        frozen = c.freeze(out, bank)
        assert c.freeze(out, bank) == frozen
        initial = out/'initializer.txt'
        original = initial.read_text()
        initial.write_text(original+' ')
        try:
            c.freeze(out, bank)
            raise AssertionError('Initializer change accepted')
        except ValueError:
            pass
        initial.write_text(original)
        cp = out/'frozen/strong_weighted.pt'
        original_bytes = cp.read_bytes()
        cp.write_bytes(b'changed')
        try:
            c.freeze(out, bank)
            raise AssertionError('Checkpoint change accepted')
        except ValueError:
            pass
        cp.write_bytes(original_bytes)
        print('PASS: protocol idempotence and frozen artifact tamper rejection', flush=True)
        # Smoke seed is separate from the preregistered confirmation seed.
        # A nonexistent default benchmark path ensures the override is actually used.
        import scripts.generate_probe_bank as generator
        load = generator.load_config
        def smoke_config():
            cfg = load()
            cfg['policy']['data_path'] = '/nonexistent/benchmark_must_not_be_opened.txt'
            return cfg
        generator.load_config = smoke_config
        try:
            generator.generate(bank, count=1, seed=98761234, stride=199, initializer_data_path=initial)
        finally:
            generator.load_config = load
        c.check_disjoint(bank, ROOT/'data/processed/probe_bank_v2')
        stamp = (bank/'instance_0000.npz').stat().st_mtime_ns
        c.generate(bank, count=1, seed=98761234, stride=199, initializer_data_path=initial)
        assert (bank/'instance_0000.npz').stat().st_mtime_ns == stamp
        result = c.evaluate(out, bank, count=1, draws=20)
        assert set(result['hypotheses']) == {'H1', 'H2', 'H3'}
        assert all(np.isfinite(m['overall']['mean_regret']) for m in result['metrics'].values())
        assert (out/'comparison.md').exists()
        assert json.loads((out/'protocol.json').read_text()) == frozen
    print('PASS: synthetic-only generation, disjointness, resume, frozen inference, final report', flush=True)
    print('ALL PHASE 2C CHECKS PASSED', flush=True)


if __name__ == '__main__':
    main()
