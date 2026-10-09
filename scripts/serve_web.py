"""Start the web UI so it can be opened from a phone: builds the frontend when needed, starts the API
(uvicorn) with an access token, optionally opens a public HTTPS tunnel (Cloudflare quick tunnel, no
account), and prints the link + a QR code to scan with the phone camera.

    python scripts/serve_web.py --checkpoint runs/large --data /path/to/datasets           # same Wi-Fi
    python scripts/serve_web.py --checkpoint runs/large --data /path/to/datasets --tunnel  # anywhere
    … --gen_root /content/drive/MyDrive/rfcav/ui_datasets    # + dataset generation from the UI
    … --checkpoint auto --ckpt_search <training_logs dir>   # newest training run there (else untrained)

Same Wi-Fi: the phone opens http://<this machine's LAN address>:<port>/?token=…
--tunnel: https://<random>.trycloudflare.com/?token=… (downloads `cloudflared` if missing; the link
lives as long as this process). Anyone holding the full link (with the token) can use the server —
share it like a password. Ctrl-C stops everything.
"""
import argparse
import glob
import hashlib
import os
import platform
import re
import secrets
import shutil
import socket
import stat
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLOUDFLARED_URL = 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-{arch}'


def lan_ip():
    """Address of this machine on the local network (the one a phone on the same Wi-Fi reaches)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))                      # no packet is sent
        return s.getsockname()[0]
    except OSError:
        return '127.0.0.1'
    finally:
        s.close()


def web_sources_hash(web=None):
    """Hash of the frontend sources (everything in web/ but node_modules and dist): dist is rebuilt when
    it changes, e.g. after a git pull."""
    web = web or os.path.join(ROOT, 'web')
    h = hashlib.sha1()
    for dp, dns, fns in os.walk(web):
        dns[:] = sorted(d for d in dns if d not in ('node_modules', 'dist'))
        for fn in sorted(f for f in fns if not f.endswith('.tsbuildinfo')):   # tsc build output
            fp = os.path.join(dp, fn)
            h.update(os.path.relpath(fp, web).encode())
            with open(fp, 'rb') as f:
                h.update(f.read())
    return h.hexdigest()


def build_frontend(force=False):
    dist = os.path.join(ROOT, 'web', 'dist', 'index.html')
    stamp = os.path.join(ROOT, 'web', 'dist', '.sources')
    want = web_sources_hash()
    have = open(stamp).read().strip() if os.path.exists(stamp) else None
    if os.path.exists(dist) and not force and (have == want or not shutil.which('npm')):
        return                                         # up to date (or prebuilt elsewhere, no npm to rebuild)
    if not shutil.which('npm'):
        sys.exit("web/dist is missing and npm is not installed: install Node.js ≥ 20.19 (or build the "
                 "frontend elsewhere: cd web && npm ci && npm run build)")
    print('building the frontend (once) …', flush=True)
    web = os.path.join(ROOT, 'web')
    subprocess.run(['npm', 'ci', '--no-audit', '--no-fund'], cwd=web, check=True)
    subprocess.run(['npm', 'run', 'build'], cwd=web, check=True)
    with open(stamp, 'w') as f:
        f.write(want)


def find_checkpoint(roots):
    """The training run (dir directly under a root, or a .ckpt file in it) holding the newest .ckpt
    below the given roots, or None."""
    best = None
    for root in roots:
        root = os.path.abspath(os.path.expanduser(root))
        for fp in glob.glob(os.path.join(root, '**', '*.ckpt'), recursive=True):
            t = os.path.getmtime(fp)
            if best is None or t > best[0]:
                rel = os.path.relpath(fp, root).split(os.sep)
                best = (t, fp if len(rel) == 1 else os.path.join(root, rel[0]))
    return best and best[1]


def cloudflared_bin():
    exe = shutil.which('cloudflared')
    if exe:
        return exe
    if platform.system() != 'Linux':
        sys.exit("cloudflared not found: install it (https://developers.cloudflare.com/cloudflare-one/"
                 "connections/connect-networks/downloads/) or run without --tunnel on the same Wi-Fi")
    arch = {'x86_64': 'amd64', 'aarch64': 'arm64'}.get(platform.machine(), 'amd64')
    dst = os.path.join(os.path.expanduser('~'), '.cache', 'rfcav', 'cloudflared')
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        print('downloading cloudflared …', flush=True)
        urllib.request.urlretrieve(CLOUDFLARED_URL.format(arch=arch), dst)
        os.chmod(dst, os.stat(dst).st_mode | stat.S_IEXEC)
    return dst


def start_tunnel(port, timeout=60):
    """(process, public https URL) of a Cloudflare quick tunnel to localhost:port."""
    p = subprocess.Popen([cloudflared_bin(), 'tunnel', '--no-autoupdate', '--url', f'http://127.0.0.1:{port}'],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    found, t0 = {}, time.time()

    def read():
        for line in p.stdout:
            m = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com', line)
            if m and 'url' not in found:
                found['url'] = m.group(0)
    threading.Thread(target=read, daemon=True).start()
    while 'url' not in found and time.time() - t0 < timeout and p.poll() is None:
        time.sleep(0.2)
    if 'url' not in found:
        p.kill()
        sys.exit('could not open the tunnel (is outbound HTTPS allowed?) — use the same-Wi-Fi link instead')
    return p, found['url']


def print_qr(url):
    try:
        import qrcode
    except ImportError:
        print('(pip install qrcode → a QR code to scan here)')
        return
    q = qrcode.QRCode(border=1)
    q.add_data(url)
    q.make(fit=True)
    q.print_ascii(invert=True)


def wait_up(port, token, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            req = urllib.request.Request(f'http://127.0.0.1:{port}/api/info', headers={'X-RFCAV-Token': token or ''})
            urllib.request.urlopen(req, timeout=2)
            return True
        except Exception:                      # noqa: BLE001 — still starting
            time.sleep(1)
    return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--checkpoint', default=None,
                    help='.ckpt or training dir; "auto": the newest run under --ckpt_search (none: UNTRAINED demo model)')
    ap.add_argument('--ckpt_search', nargs='*', default=[], help='where --checkpoint auto looks (e.g. training_logs)')
    ap.add_argument('--checkpoint_optional', action='store_true',
                    help='a checkpoint that does not load → untrained model + a warning (instead of an error)')
    ap.add_argument('--data', nargs='*', default=[], help='dataset roots (PKL files / H5 directories)')
    ap.add_argument('--gen_root', default=None,
                    help='folder for datasets generated from the UI (e.g. a Google Drive folder); off when unset')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--device', default=None, help='cpu | cuda (default: cuda when available)')
    ap.add_argument('--tunnel', action='store_true', help='public HTTPS link (Cloudflare quick tunnel)')
    ap.add_argument('--no_token', action='store_true', help='no access token (only on a trusted network)')
    ap.add_argument('--rebuild', action='store_true', help='rebuild the frontend')
    args = ap.parse_args(argv)

    build_frontend(args.rebuild)
    ckpt = args.checkpoint
    if ckpt == 'auto':
        ckpt = find_checkpoint(args.ckpt_search)
        print(f"checkpoint: {ckpt or 'none found → untrained demo model (generation and the dataset view work)'}",
              flush=True)
        args.checkpoint_optional = True
    token = None if args.no_token else secrets.token_urlsafe(12)
    env = dict(os.environ, RFCAV_DATA=os.pathsep.join(os.path.abspath(d) for d in args.data))
    gen = os.path.abspath(args.gen_root) if args.gen_root else None
    if gen:
        os.makedirs(gen, exist_ok=True)
    for k, v in (('RFCAV_CHECKPOINT', ckpt), ('RFCAV_DEVICE', args.device), ('RFCAV_TOKEN', token),
                 ('RFCAV_GEN_ROOT', gen), ('RFCAV_CHECKPOINT_OPTIONAL', '1' if args.checkpoint_optional else None)):
        if v:
            env[k] = v
        else:
            env.pop(k, None)
    host = '127.0.0.1' if args.tunnel else '0.0.0.0'
    server = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'src.service.api:app', '--host', host,
                               '--port', str(args.port)], cwd=ROOT, env=env)
    procs = [server]
    try:
        if not wait_up(args.port, token):
            sys.exit('the server did not start (see the log above)')
        q = f'/?token={token}' if token else '/'
        if args.tunnel:
            tp, base = start_tunnel(args.port)
            procs.append(tp)
            url = base + q
        else:
            url = f'http://{lan_ip()}:{args.port}{q}'
        print('\n' + '=' * 72)
        print(f'  Telefondan açın / open on the phone:\n  {url}')
        print('=' * 72, flush=True)
        print_qr(url)
        if not args.tunnel:
            print('(telefon aynı Wi-Fi ağında olmalı; olmuyorsa --tunnel kullanın)')
        print('durdurmak için Ctrl-C', flush=True)
        server.wait()
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()


if __name__ == '__main__':
    main()
