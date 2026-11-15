#!/usr/bin/env python3
"""Run a script or a named stage from the release root, whatever the current directory.
  python3 run.py verify            check headline numbers against the bundled tables
  python3 run.py figures           redraw Figs 1-5 from the bundled tables
  python3 run.py pilot/NAME.py [args]   run one analysis script"""
import os, sys, subprocess
root=os.path.dirname(os.path.abspath(__file__)); os.chdir(root); os.makedirs('figs',exist_ok=True)
stages={'verify':[['verify.py']],'figures':[['pilot/figs_v6.py'],['pilot/fig45.py']]}
a=sys.argv[1:] or ['verify']
for cmd in stages.get(a[0],[a]):
    print('>>',' '.join(cmd),flush=True); rc=subprocess.call([sys.executable]+cmd)
    if rc: sys.exit(rc)
