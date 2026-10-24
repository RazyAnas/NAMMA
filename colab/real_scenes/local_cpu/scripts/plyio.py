import numpy as np, zipfile
def read(src, member=None):
    f = zipfile.ZipFile(src).open(member) if member else open(src, 'rb')
    hdr = b''
    while True:
        l = f.readline(); hdr += l
        if l.strip() == b'end_header': break
    lines = hdr.decode().splitlines()
    n = int([l for l in lines if l.startswith('element vertex')][0].split()[-1])
    names = [l.split()[-1] for l in lines if l.startswith('property')]
    a = np.frombuffer(f.read(n * 4 * len(names)), dtype=np.float32).reshape(n, len(names)).copy()
    return hdr, names, a
def write(path, hdr, names, a):
    lines = hdr.decode().splitlines(keepends=True)
    lines = [f'element vertex {len(a)}\n' if l.startswith('element vertex') else l for l in lines]
    with open(path, 'wb') as f:
        f.write(''.join(lines).encode()); f.write(a.astype(np.float32).tobytes())
