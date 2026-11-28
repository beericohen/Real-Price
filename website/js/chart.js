/* The price-curve chart, drawn as plain SVG (no chart library).
   It shows the model's price for the user's property while ONE parameter (area or year built) is changed,
   with the 10%-90% range as a band and the user's own value highlighted. */
(function () {
  const SVG_NS = "http://www.w3.org/2000/svg";
  const WIDTH = 900;
  const HEIGHT = 460;
  const PAD = { top: 24, right: 28, bottom: 52, left: 74 }; // room for the axis labels
  const X_AXIS_TITLES = { area: "שטח (מ״ר)", year: "שנת בנייה" };

  /** Creates an SVG element with the given attributes. */
  function svgElement(tag, attributes = {}) {
    const element = document.createElementNS(SVG_NS, tag);
    for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, value);
    return element;
  }

  function svgText(text, attributes) {
    const element = svgElement("text", attributes);
    element.textContent = text;
    return element;
  }

  /** 3,250,000 -> "₪3.3M", 450,000 -> "₪450K" (short labels for the y axis). */
  function shortMoney(n) {
    if (n >= 1e6) return "₪" + (n / 1e6).toFixed(n >= 1e7 ? 0 : 1) + "M";
    return "₪" + Math.round(n / 1000) + "K";
  }

  class CurveChart {
    constructor(svg) {
      this.svg = svg;
      this.axis = "area"; // which curve is shown: "area" or "year"
      this.data = null; // {area: {pts, you}, year: {pts, you}}, set by setData()
      svg.setAttribute("viewBox", `0 0 ${WIDTH} ${HEIGHT}`);
      this.draw();
    }

    setAxis(axis) {
      this.axis = axis;
      this.draw();
    }

    setData(data) {
      this.data = data;
      this.draw();
    }

    draw() {
      const svg = this.svg;
      while (svg.firstChild) svg.removeChild(svg.firstChild); // clear the old drawing

      const curve = this.data && this.data[this.axis];
      const plotWidth = WIDTH - PAD.left - PAD.right;
      const plotHeight = HEIGHT - PAD.top - PAD.bottom;

      if (!curve) {
        svg.append(svgText("הריצו הערכה כדי לראות את העקום", { x: WIDTH / 2, y: HEIGHT / 2, "text-anchor": "middle", class: "axis-title" }));
        return;
      }

      // ---- scales: data value -> pixel position
      const xValues = curve.pts.map((p) => p.x);
      const xMin = Math.min(...xValues);
      const xMax = Math.max(...xValues);
      const yMax = Math.max(...curve.pts.map((p) => p.hi), curve.you ? curve.you.p : 0) * 1.08; // 8% headroom
      const scaleX = (v) => PAD.left + ((v - xMin) / (xMax - xMin || 1)) * plotWidth;
      const scaleY = (v) => PAD.top + plotHeight - (v / yMax) * plotHeight;

      // ---- y axis: 5 grid lines with price labels
      for (let i = 0; i < 5; i++) {
        const y = PAD.top + plotHeight - (i * plotHeight) / 4;
        svg.append(svgElement("line", { x1: PAD.left, x2: PAD.left + plotWidth, y1: y, y2: y, class: "gridline" }));
        svg.append(svgText(shortMoney((yMax * i) / 4), { x: PAD.left - 12, y: y + 4, "text-anchor": "end", class: "axis-text" }));
      }
      // ---- x axis: 6 labels and the axis title
      for (let i = 0; i < 6; i++) {
        const v = xMin + ((xMax - xMin) * i) / 5;
        svg.append(svgText(Math.round(v), { x: scaleX(v), y: HEIGHT - PAD.bottom + 25, "text-anchor": "middle", class: "axis-text" }));
      }
      svg.append(svgText(X_AXIS_TITLES[this.axis], { x: PAD.left + plotWidth / 2, y: HEIGHT - 10, "text-anchor": "middle", class: "axis-title" }));

      // ---- the curves. "M" = move to the first point, "L" = line to the next one.
      const linePath = (key) => curve.pts.map((p, i) => `${i ? "L" : "M"}${scaleX(p.x).toFixed(1)},${scaleY(p[key]).toFixed(1)}`).join("");
      // The band between the low and high curves: along the top edge, then back along the bottom edge, then close
      const lowEdgeBackwards = curve.pts.map((p) => `L${scaleX(p.x).toFixed(1)},${scaleY(p.lo).toFixed(1)}`).reverse().join("");
      svg.append(svgElement("path", { d: linePath("hi") + lowEdgeBackwards + "Z", class: "curve-band" }));
      svg.append(svgElement("path", { d: linePath("hi"), class: "curve-edge" }), svgElement("path", { d: linePath("lo"), class: "curve-edge" }));
      svg.append(svgElement("path", { d: linePath("p"), class: "curve-line" }));

      // ---- the user's own property
      if (curve.you) {
        const x = scaleX(curve.you.x);
        const y = scaleY(curve.you.p);
        svg.append(
          svgElement("circle", { cx: x, cy: y, r: 15, class: "you-halo" }),
          svgElement("circle", { cx: x, cy: y, r: 6, class: "you-point" }),
          svgText("הנכס שלכם", { x, y: y - 20, "text-anchor": "middle", class: "you-label" })
        );
      } else {
        svg.append(svgText("שנת הבנייה לא צוינה", { x: PAD.left + plotWidth - 4, y: PAD.top + 14, "text-anchor": "end", class: "axis-title" }));
      }
    }
  }

  RP.CurveChart = CurveChart;
})();
