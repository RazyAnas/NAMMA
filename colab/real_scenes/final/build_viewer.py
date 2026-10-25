"""cloud.npz + walk.npz -> viewer.html: one self-contained file (raw WebGL, no CDN). Walk with WASD + drag; movement is blocked
by the walkable grid (collision), minimap shows the plan. Usage: python build_viewer.py <model_dir>"""
import base64, json, sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "experiments" / "nav_from_mesh"))
import nav_from_mesh as N

def build(model, max_pts=400000):
    model = pathlib.Path(model); c = np.load(model / "cloud.npz"); w = np.load(model / "walk.npz")
    up = w["up"]; t, b = N.floor_frame(up); u0, v0, cell = w["grid"]
    pts = c["pts"]; h = pts @ up - float(w["floor"])
    xyz = np.stack([pts @ t, h, pts @ b], 1).astype(np.float32)
    keep = (h > -0.3) & (h < 3.0)                                   # drop what is far below the floor / above the ceiling
    xyz, col = xyz[keep], c["col"][keep]
    if len(xyz) > max_pts: i = np.random.default_rng(0).choice(len(xyz), max_pts, replace=False); xyz, col = xyz[i], col[i]
    walk = w["walk"]; idx = np.argwhere(walk); start = idx[len(idx) // 2]
    meta = dict(n=len(xyz), W=int(walk.shape[0]), H=int(walk.shape[1]), u0=float(u0), v0=float(v0), cell=float(cell),
                start=[float(u0 + (start[0] + .5) * cell), float(v0 + (start[1] + .5) * cell)])
    html = (TEMPLATE.replace("__META__", json.dumps(meta)).replace("__PTS__", base64.b64encode(xyz.tobytes()).decode())
            .replace("__COL__", base64.b64encode(col.tobytes()).decode()).replace("__WALK__", base64.b64encode(np.packbits(walk.astype(np.uint8).ravel()).tobytes()).decode()))
    (model / "viewer.html").write_text(html); return meta

TEMPLATE = r"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>NAMMA walk</title>
<style>html,body{margin:0;height:100%;background:#111;overflow:hidden;font:13px system-ui;color:#ddd}canvas#c{width:100%;height:100%;display:block;touch-action:none}
#mm{position:fixed;right:10px;bottom:10px;border:1px solid #555;background:#000a;width:200px}#hud{position:fixed;left:10px;top:10px;background:#000a;padding:6px 10px;border-radius:6px}</style>
<canvas id=c></canvas><canvas id=mm></canvas><div id=hud>drag = look · WASD / arrows = walk (collision on) · Shift = fast · G = ghost (no collision)</div>
<script>
const M=__META__,b64=s=>Uint8Array.from(atob(s),c=>c.charCodeAt(0));
const P=new Float32Array(b64("__PTS__").buffer),C=b64("__COL__"),pk=b64("__WALK__");
const walk=new Uint8Array(M.W*M.H);for(let i=0;i<walk.length;i++)walk[i]=(pk[i>>3]>>(7-(i&7)))&1;
const ok=(x,z)=>{const i=Math.floor((x-M.u0)/M.cell),j=Math.floor((z-M.v0)/M.cell);return i>=0&&j>=0&&i<M.W&&j<M.H&&walk[i*M.H+j]};
const cv=document.getElementById('c'),gl=cv.getContext('webgl');
const sh=(t,s)=>{const o=gl.createShader(t);gl.shaderSource(o,s);gl.compileShader(o);return o};
const pr=gl.createProgram();gl.attachShader(pr,sh(gl.VERTEX_SHADER,`attribute vec3 p;attribute vec3 c;uniform mat4 m;uniform float s;varying vec3 v;void main(){gl_Position=m*vec4(p,1);v=c;gl_PointSize=clamp(s/gl_Position.w,1.5,7.);}`));
gl.attachShader(pr,sh(gl.FRAGMENT_SHADER,`precision mediump float;varying vec3 v;void main(){gl_FragColor=vec4(v,1);}`));gl.linkProgram(pr);gl.useProgram(pr);
const buf=(d,n,l)=>{const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,d,gl.STATIC_DRAW);const a=gl.getAttribLocation(pr,l);gl.enableVertexAttribArray(a);gl.vertexAttribPointer(a,3,n,n==gl.UNSIGNED_BYTE,0,0)};
buf(P,gl.FLOAT,'p');buf(C,gl.UNSIGNED_BYTE,'c');
let x=M.start[0],z=M.start[1],yaw=0,pitch=0,ghost=false;const EYE=1.5,keys={};
onkeydown=e=>{keys[e.key.toLowerCase()]=1;if(e.key=='g'||e.key=='G')ghost=!ghost};onkeyup=e=>keys[e.key.toLowerCase()]=0;
let drag=0,lx,ly;cv.onpointerdown=e=>{drag=1;lx=e.clientX;ly=e.clientY};onpointerup=()=>drag=0;
onpointermove=e=>{if(!drag)return;yaw-=(e.clientX-lx)*.004;pitch=Math.max(-1.4,Math.min(1.4,pitch-(e.clientY-ly)*.004));lx=e.clientX;ly=e.clientY};
function mat(){const a=cv.width/cv.height,f=1/Math.tan(.5),n=.05,fa=100,cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
const fw=[-sy*cp,sp,-cy*cp],r=[cy,0,-sy],u=[sy*sp,cp,cy*sp];const e=[x,EYE,z];const d=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const V=[r[0],u[0],-fw[0],0,r[1],u[1],-fw[1],0,r[2],u[2],-fw[2],0,-d(r,e),-d(u,e),d(fw,e),1];
const Pm=[f/a,0,0,0,0,f,0,0,0,0,(fa+n)/(n-fa),-1,0,0,2*fa*n/(n-fa),0];const o=new Float32Array(16);
for(let i=0;i<4;i++)for(let j=0;j<4;j++){let s=0;for(let k=0;k<4;k++)s+=Pm[k*4+j]*V[i*4+k];o[i*4+j]=s}return o}
const mm=document.getElementById('mm'),mc=mm.getContext('2d');mm.width=200;mm.height=Math.round(200*M.H/M.W);
const im=mc.createImageData(M.W,M.H);const off=document.createElement('canvas');off.width=M.W;off.height=M.H;
for(let i=0;i<M.W;i++)for(let j=0;j<M.H;j++){const k=(j*M.W+i)*4,v=walk[i*M.H+j];im.data[k]=v?60:15;im.data[k+1]=v?170:15;im.data[k+2]=v?100:20;im.data[k+3]=255}
off.getContext('2d').putImageData(im,0,0);
function frame(t){cv.width=innerWidth;cv.height=innerHeight;gl.viewport(0,0,cv.width,cv.height);gl.enable(gl.DEPTH_TEST);gl.clearColor(.07,.07,.08,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
const sp=(keys.shift?2.5:1.2)/60,fw=(keys.w||keys.arrowup?1:0)-(keys.s||keys.arrowdown?1:0),st=(keys.d||keys.arrowright?1:0)-(keys.a||keys.arrowleft?1:0);
const dx=(-Math.sin(yaw)*fw+Math.cos(yaw)*st)*sp,dz=(-Math.cos(yaw)*fw-Math.sin(yaw)*st)*sp;
if(ghost||ok(x+dx,z))x+=dx;if(ghost||ok(x,z+dz))z+=dz;
gl.uniformMatrix4fv(gl.getUniformLocation(pr,'m'),false,mat());gl.uniform1f(gl.getUniformLocation(pr,'s'),cv.height*.012);gl.drawArrays(gl.POINTS,0,M.n);
mc.drawImage(off,0,0,mm.width,mm.height);mc.fillStyle='#f55';mc.beginPath();mc.arc((x-M.u0)/M.cell/M.W*mm.width,(z-M.v0)/M.cell/M.H*mm.height,4,0,7);mc.fill();requestAnimationFrame(frame)}
frame();
</script>"""
if __name__ == "__main__": print(build(sys.argv[1]))
