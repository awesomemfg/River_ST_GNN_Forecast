"""Use the shared chronological trainer for the nodewise LSTM.
Evaluated epochs: seed 101 = 19, seed 202 = 12, seed 303 = 21.
The random-validation trainer is retained only in archive/model_lineage.
"""
from pathlib import Path
import runpy
import sys

trainer = Path(__file__).resolve().parents[1] / 'stgnn' / 'train.py'
if '--model-kind' in sys.argv:
    raise ValueError('This entry point selects lstm. Do not also pass --model-kind.')
sys.path.insert(0, str(trainer.parent))
sys.argv = [str(trainer), '--model-kind', 'lstm'] + sys.argv[1:]
print('Shared chronological LSTM trainer:', trainer, flush=True)
runpy.run_path(str(trainer), run_name='__main__')
