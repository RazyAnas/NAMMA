Vendored so the phone page works on networks that block CDNs (campus Wi-Fi) and offline after load.

| File | Package | Version | Licence |
|---|---|---|---|
| three.module.min.js, three.core.min.js | three | 0.180.0 | MIT (LICENSE-three.txt) |
| addons/controls/OrbitControls.js, addons/postprocessing/Pass.js | three (examples/jsm) | 0.180.0 | MIT |
| spark.module.min.js | @sparkjsdev/spark (World Labs, Gaussian splat renderer for three.js) | 2.3.1 | MIT (LICENSE-spark.txt) |

Update: `npm pack three@<v> @sparkjsdev/spark@<v>` and copy the same files; Spark 2.3 needs three >= 0.180.
