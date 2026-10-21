# Make DN-Splatter (written for gsplat 1.0) run on gsplat 1.4, which dropped the legacy rasterizer:
#  - the legacy imports become fallbacks (quat_to_rotmat and num_sh_bases are re-implemented in torch if missing)
#  - the per-Gaussian normal image, drawn with the legacy rasterize_gaussians, is drawn with the current rasterization()
#    using the normals as colours, same inputs as the main RGB+depth render.
# Usage: python patch_dn_splatter.py <dn-splatter checkout>
import sys
from pathlib import Path
p = Path(sys.argv[1]) / 'dn_splatter/dn_model.py'; s = p.read_text()
if 'PATCHED_FOR_GSPLAT_1_4' in s: print('already patched'); sys.exit(0)
s = s.replace('''from gsplat import rasterize_gaussians
from gsplat.cuda_legacy._torch_impl import quat_to_rotmat
from gsplat.cuda_legacy._wrapper import num_sh_bases
''', '''# PATCHED_FOR_GSPLAT_1_4
try:
    from gsplat.cuda_legacy._torch_impl import quat_to_rotmat
except ImportError:
    def quat_to_rotmat(quat):
        w, x, y, z = torch.unbind(F.normalize(quat, dim=-1), dim=-1)
        return torch.stack([1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y),
                            2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x),
                            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)], dim=-1).reshape(quat.shape[:-1] + (3, 3))
try:
    from gsplat.cuda_legacy._wrapper import num_sh_bases
except ImportError:
    def num_sh_bases(degree):
        return (degree + 1) ** 2
''')
old_start = s.index('            normals_im: Tensor = rasterize_gaussians(')
old_end = s.index('            # convert normals from [-1,1] to [0,1]', old_start)
s = s[:old_start] + '''            normals_im, _, _ = rasterization(
                means=means_crop,
                quats=quats_crop,
                scales=torch.exp(scales_crop),
                opacities=torch.sigmoid(opacities_crop).squeeze(-1),
                colors=normals[None],
                viewmats=viewmat,
                Ks=K,
                width=W,
                height=H,
                tile_size=BLOCK_WIDTH,
                packed=False,
                near_plane=0.01,
                far_plane=1e10,
                render_mode="RGB",
                sh_degree=None,
                rasterize_mode=self.config.rasterize_mode,
            )
            normals_im = normals_im[0]
''' + s[old_end:]
assert 'rasterize_gaussians(' not in s
p.write_text(s); print('patched', p)
