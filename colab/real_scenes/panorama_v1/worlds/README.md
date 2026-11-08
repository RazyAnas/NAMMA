# World requests

Pushing `<tag>.json` here (on `feature/streetview-panorama-3d`) runs `.github/workflows/namma-street-world.yml`:
it builds the browser game for one exported world and publishes it at
`https://anasrazy-namma-street.static.hf.space/world/<tag>/index.html
(the `anasrazy/namma-street` Space; `namma-tour` reached its 1 GB storage limit)`.

```json
{"tag": "paris-h7", "kernel": "anasrazy/namma-pano-exp-h7b", "experiment": "H7-full", "lod_quality": true}
```
Leave `kernel` empty to publish the synthetic test world. No GPU training is started.
