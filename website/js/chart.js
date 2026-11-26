/* Price-curve chart (plain SVG): the model's price for the user's property while one parameter sweeps; the user's value is highlighted. */
(function () {
  const NS = "http://www.w3.org/2000/svg", W = 900, H = 460, P = { t: 24, r: 28, b: 52, l: 74 };
  const E = (tag, a = {}) => { const n = document.createElementNS(NS, tag); Object.entries(a).forEach(([k, v]) => n.setAttribute(k, v)); return n; };
  const short = (n) => "₪" + (n >= 1e6 ? (n / 1e6).toFixed(n >= 1e7 ? 0 : 1) + "M" : Math.round(n / 1000) + "K");

  class CurveChart {
    constructor(svg) { this.svg = svg; this.axis = "area"; this.data = null; svg.setAttribute("viewBox", `0 0 ${W} ${H}`); this.render(); }
    setAxis(a) { this.axis = a; this.render(); }
    setData(d) { this.data = d; this.render(); }

    render() {
      const svg = this.svg;
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      const d = this.data && this.data[this.axis];
      const iw = W - P.l - P.r, ih = H - P.t - P.b;
      if (!d) { const t = E("text", { x: W / 2, y: H / 2, "text-anchor": "middle", class: "axis-title" }); t.textContent = "הריצו הערכה כדי לראות את העקום"; svg.append(t); return; }

      const xs = d.pts.map((p) => p.x), xmin = Math.min(...xs), xmax = Math.max(...xs);
      const ymax = Math.max(...d.pts.map((p) => p.hi), d.you ? d.you.p : 0) * 1.08;
      const sx = (v) => P.l + ((v - xmin) / (xmax - xmin || 1)) * iw, sy = (v) => P.t + ih - (v / ymax) * ih;

      for (let i = 0; i < 5; i++) {
        const y = P.t + ih - (i * ih) / 4;
        svg.append(E("line", { x1: P.l, x2: P.l + iw, y1: y, y2: y, class: "gridline" }));
        const t = E("text", { x: P.l - 12, y: y + 4, "text-anchor": "end", class: "axis-text" }); t.textContent = short((ymax * i) / 4); svg.append(t);
      }
      for (let i = 0; i < 6; i++) {
        const v = xmin + ((xmax - xmin) * i) / 5, t = E("text", { x: sx(v), y: H - P.b + 25, "text-anchor": "middle", class: "axis-text" });
        t.textContent = Math.round(v); svg.append(t);
      }
      const xl = E("text", { x: P.l + iw / 2, y: H - 10, "text-anchor": "middle", class: "axis-title" });
      xl.textContent = this.axis === "area" ? "שטח (מ״ר)" : "שנת בנייה"; svg.append(xl);

      const line = (key) => d.pts.map((p, i) => `${i ? "L" : "M"}${sx(p.x).toFixed(1)},${sy(p[key]).toFixed(1)}`).join("");
      const lo = d.pts.map((p) => `L${sx(p.x).toFixed(1)},${sy(p.lo).toFixed(1)}`).reverse().join("");
      svg.append(E("path", { d: line("hi") + lo + "Z", class: "curve-band" }));
      svg.append(E("path", { d: line("hi"), class: "curve-edge" }), E("path", { d: line("lo"), class: "curve-edge" }));
      svg.append(E("path", { d: line("p"), class: "curve-line" }));

      if (d.you) {
        const x = sx(d.you.x), y = sy(d.you.p);
        const lab = E("text", { x, y: y - 20, "text-anchor": "middle", class: "you-label" }); lab.textContent = "הנכס שלכם";
        svg.append(E("circle", { cx: x, cy: y, r: 15, class: "you-halo" }), E("circle", { cx: x, cy: y, r: 6, class: "you-point" }), lab);
      } else {
        const t = E("text", { x: P.l + iw - 4, y: P.t + 14, "text-anchor": "end", class: "axis-title" }); t.textContent = "שנת הבנייה לא צוינה"; svg.append(t);
      }
    }
  }
  window.CurveChart = CurveChart;
})();
