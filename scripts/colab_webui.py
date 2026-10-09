"""Everything the web UI needs on a fresh Colab VM, then the server (scripts/serve_web.py) — called by the
one-cell notebook Colab_WebUI.ipynb after it has mounted Drive and cloned the repo:

    system libraries for gmsh · pip requirements · Node.js ≥ 22 · frontend build (web/dist, rebuilt when
    web/ changes) · Drive folders · checkpoint (newest run under <drive>/training_logs, else the untrained
    model) · server + Cloudflare tunnel + QR code

Each step is skipped when already done, so re-running the cell on the same VM starts in seconds.
Dataset generation runs here, on the Colab VM's CPUs (one worker per core), and writes to
<gen_dir>/<TAG>/h5/<family>/… on Drive; training (the "Eğitim" page) runs train.py on the VM's GPU
and writes checkpoints to <drive_dir>/training_logs/<run>/.

    python scripts/colab_webui.py --drive_dir /content/drive/MyDrive/rf_cavity_3d --tunnel
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
APT = ['libglu1-mesa', 'libxrender1', 'libxcursor1', 'libxft2', 'libxinerama1']   # gmsh (requirements.txt)
NODE_MAJOR = 22
STAMPS = os.path.join(os.path.expanduser('~'), '.cache', 'rfcav')


def step(msg):
    print(f"\n── {msg}", flush=True)


def sh(cmd, **kw):
    subprocess.run(cmd, shell=isinstance(cmd, str), check=True, **kw)


def apt_libs():
    missing = [p for p in APT
               if subprocess.run(['dpkg', '-s', p], capture_output=True).returncode != 0]
    if not missing:
        return print('system libraries: ok')
    step(f"apt: {' '.join(missing)}")
    sh('apt-get -qq update > /dev/null && apt-get -qq install -y ' + ' '.join(missing) + ' > /dev/null')


def pip_requirements():
    reqs = [os.path.join(ROOT, f) for f in ('requirements.txt', 'requirements-web.txt')]
    h = hashlib.sha1(b''.join(open(r, 'rb').read() for r in reqs) + sys.executable.encode()).hexdigest()[:12]
    stamp = os.path.join(STAMPS, f'pip_{h}')
    if os.path.exists(stamp):
        return print('python packages: ok')
    step('pip install -r requirements.txt -r requirements-web.txt')
    sh([sys.executable, '-m', 'pip', 'install', '-q', *sum((['-r', r] for r in reqs), [])])
    os.makedirs(STAMPS, exist_ok=True)
    open(stamp, 'w').close()


def node_major():
    exe = shutil.which('node')
    if not exe:
        return 0
    v = subprocess.run([exe, '--version'], capture_output=True, text=True).stdout.strip()
    try:
        return int(v.lstrip('v').split('.')[0])
    except ValueError:
        return 0


def node():
    if node_major() >= NODE_MAJOR and shutil.which('npm'):
        return print(f'node: v{node_major()} ok')
    step(f'Node.js {NODE_MAJOR} (NodeSource)')
    sh(f'curl -fsSL https://deb.nodesource.com/setup_{NODE_MAJOR}.x | bash - > /dev/null 2>&1 '
       '&& apt-get -qq install -y nodejs > /dev/null')
    if node_major() < NODE_MAJOR:
        sys.exit('Node.js installation failed (see above)')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--drive_dir', default='/content/drive/MyDrive/rf_cavity_3d',
                    help='Drive folder: datasets (PKL / H5) and training_logs/ of the notebooks')
    ap.add_argument('--gen_dir', default=None, help='generated datasets (default: <drive_dir>/ui_datasets)')
    ap.add_argument('--checkpoint', default='auto',
                    help='"auto" (newest run under <drive_dir>/training_logs), "" (untrained) or a path')
    ap.add_argument('--tunnel', action='store_true', help='public HTTPS link + QR code (phone)')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--skip_install', action='store_true', help='no apt / pip / node (already set up)')
    args = ap.parse_args(argv)

    drive = os.path.abspath(args.drive_dir)
    gen = os.path.abspath(args.gen_dir or os.path.join(drive, 'ui_datasets'))
    if not args.skip_install:
        step('setup')
        apt_libs()
        pip_requirements()
        node()
    os.makedirs(gen, exist_ok=True)
    print(f'Drive: {drive}\n  generated datasets → {gen}\n  training runs      → {drive}/training_logs')

    import serve_web
    runs = os.path.join(drive, 'training_logs')
    cmd = ['--data', drive, '--gen_root', gen, '--runs_root', runs, '--port', str(args.port), '--checkpoint_optional']
    if args.checkpoint == 'auto':
        cmd += ['--checkpoint', 'auto', '--ckpt_search', runs]
    elif args.checkpoint:
        cmd += ['--checkpoint', args.checkpoint]
    if args.tunnel:
        cmd.append('--tunnel')
    step(f'server ({os.cpu_count()} CPU cores for dataset generation)')
    serve_web.main(cmd)


if __name__ == '__main__':
    main()
