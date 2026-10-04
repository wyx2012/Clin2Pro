"""Run existing assertion-based tests without installing pytest."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests import test_core

if __name__ == '__main__':
    tests = [(name, fn) for name, fn in vars(test_core).items()
             if name.startswith('test_') and callable(fn)]
    for name, fn in tests:
        fn()
        print('PASS', name)
    print('%d core tests passed' % len(tests))
