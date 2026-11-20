// Visualizador 3D mínimo em canvas 2D (sem dependências, funciona offline no canteiro).
// Arrastar = girar; roda = zoom. Z para cima.
class Viewer {
  constructor(canvas) {
    this.c = canvas; this.ctx = canvas.getContext("2d");
    this.yaw = -0.7; this.pitch = 0.35; this.zoom = 1; this.pts = null; this.corners = null;
    let drag = null;
    canvas.onpointerdown = e => { drag = [e.clientX, e.clientY]; canvas.setPointerCapture(e.pointerId); };
    canvas.onpointermove = e => {
      if (!drag) return;
      this.yaw += (e.clientX - drag[0]) * 0.01;
      this.pitch = Math.max(-1.5, Math.min(1.5, this.pitch + (e.clientY - drag[1]) * 0.01));
      drag = [e.clientX, e.clientY]; this.draw();
    };
    canvas.onpointerup = () => { drag = null; };
    canvas.onwheel = e => { e.preventDefault(); this.zoom *= e.deltaY < 0 ? 1.1 : 0.9; this.draw(); };
    new ResizeObserver(() => this.draw()).observe(canvas.parentElement);
  }
  // points: array plano [x,y,z,...]; corners: 4 pontos (12 números) do retângulo de referência
  set(points, corners = null) {
    this.pts = Float32Array.from(points); this.corners = corners;
    const all = corners ? points.concat(corners.flat()) : points;
    const lo = [1e9, 1e9, 1e9], hi = [-1e9, -1e9, -1e9];
    for (let i = 0; i < all.length; i += 3) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], all[i + k]); hi[k] = Math.max(hi[k], all[i + k]); }
    this.center = lo.map((v, k) => (v + hi[k]) / 2);
    this.radius = Math.max(...hi.map((v, k) => v - lo[k])) / 2 || 1;
    this.draw();
  }
  proj(x, y, z) {
    x -= this.center[0]; y -= this.center[1]; z -= this.center[2];
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw), cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const x1 = x * cy - y * sy, y1 = x * sy + y * cy;
    return [x1, z * cp - y1 * sp];
  }
  draw() {
    const c = this.c, dpr = window.devicePixelRatio || 1, W = c.parentElement.clientWidth, H = c.parentElement.clientHeight;
    if (c.width !== W * dpr || c.height !== H * dpr) { c.width = W * dpr; c.height = H * dpr; }
    const g = this.ctx; g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
    if (!this.pts) { g.fillStyle = getComputedStyle(document.body).color; g.globalAlpha = .6;
      g.fillText("Carregue uma nuvem de pontos para visualizar.", 20, 30); g.globalAlpha = 1; return; }
    const s = Math.min(W, H) / (2.4 * this.radius) * this.zoom, ox = W / 2, oy = H / 2;
    g.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#0b6e80";
    for (let i = 0; i < this.pts.length / 3; i++) {
      const [px, py] = this.proj(this.pts[3 * i], this.pts[3 * i + 1], this.pts[3 * i + 2]);
      g.fillRect(ox + px * s, oy - py * s, 2, 2);
    }
    if (this.corners) {
      g.strokeStyle = getComputedStyle(document.body).color; g.lineWidth = 1.5; g.setLineDash([6, 4]); g.beginPath();
      this.corners.concat([this.corners[0]]).forEach((p, i) => { const [px, py] = this.proj(...p); i ? g.lineTo(ox + px * s, oy - py * s) : g.moveTo(ox + px * s, oy - py * s); });
      g.stroke(); g.setLineDash([]);
    }
  }
}
