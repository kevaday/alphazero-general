import os
import sys
import shutil


if len(sys.argv) > 1:
    run_name = sys.argv[1]
else:
    print('Please provide a run name as a command line argument to remove training files.')
    sys.exit()

for run_dir in ('checkpoint', 'data', 'runs'):
    shutil.rmtree(os.path.join(run_dir, run_name), onerror=lambda _, path, exc_info: print(f'Failed to remove path {path}: {exc_info}'))
