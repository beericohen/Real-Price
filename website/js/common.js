/* Shared helpers: formatting, geo maths, spatial grid index, lazy parcel loading, geocoding, map factory. */
(function () {
  const RP = (window.RP = window.RP || {});
  const E5 = 1e5; // coordinates in the data files are integers = degrees * 1e5

  // ---------------------------------------------------------------- formatting
  const nf = new Intl.NumberFormat("he-IL");
  RP.fmt = (n) => nf.format(Math.round(n));
  RP.money = (n) => "₪" + RP.fmt(n);
  RP.meters = (m) => (m >= 1000 ? (m / 1000).toFixed(2) + " ק״מ" : RP.fmt(m) + " מ׳");

  // ---------------------------------------------------------------- geo
  RP.haversine = (lat1, lon1, lat2, lon2) => {
    const rad = Math.PI / 180, dLat = (lat2 - lat1) * rad, dLon = (lon2 - lon1) * rad;
    const a = Math.sin(dLat / 2) ** 2 + Math.cos(lat1 * rad) * Math.cos(lat2 * rad) * Math.sin(dLon / 2) ** 2;
    return 12742000 * Math.asin(Math.sqrt(a)); // metres
  };

  /** Uniform grid over lat/lon: nearest / radius queries only look at the few cells around the query point. */
  class GridIndex {
    constructor(points, cell = 0.01) {
      this.cell = cell; this.pts = points; this.cells = new Map();
      this.minCellM = cell * 111320 * Math.cos((33.4 * Math.PI) / 180); // shortest cell side inside Israel (metres)
      points.forEach((p, i) => {
        const key = this.key(Math.floor(p.lat / cell), Math.floor(p.lon / cell));
        if (!this.cells.has(key)) this.cells.set(key, []);
        this.cells.get(key).push(i);
      });
    }
    key(i, j) { return i + "," + j; }
    /** Visit points of all cells at Chebyshev ring distance exactly r around (ci, cj). */
    *ring(ci, cj, r) {
      for (let di = -r; di <= r; di++)
        for (let dj = -r; dj <= r; dj++) {
          if (Math.max(Math.abs(di), Math.abs(dj)) !== r) continue;
          for (const idx of this.cells.get(this.key(ci + di, cj + dj)) || []) yield this.pts[idx];
        }
    }
    /** Nearest point: grow rings until the best distance is provably smaller than anything in the next ring. */
    nearest(lat, lon, maxRings = 60) {
      const ci = Math.floor(lat / this.cell), cj = Math.floor(lon / this.cell);
      let best = null;
      for (let r = 0; r <= maxRings; r++) {
        for (const p of this.ring(ci, cj, r)) {
          const d = RP.haversine(lat, lon, p.lat, p.lon);
          if (!best || d < best.d) best = { d, p };
        }
        if (best && best.d <= r * this.minCellM) break;
      }
      return best;
    }
    countWithin(lat, lon, radius) {
      const ci = Math.floor(lat / this.cell), cj = Math.floor(lon / this.cell), rings = Math.ceil(radius / this.minCellM);
      let n = 0;
      for (let r = 0; r <= rings; r++)
        for (const p of this.ring(ci, cj, r)) if (RP.haversine(lat, lon, p.lat, p.lon) <= radius) n++;
      return n;
    }
  }
  RP.GridIndex = GridIndex;

  // ---------------------------------------------------------------- points of interest
  const indexes = {};
  /** Nearest school / transit stop + counts for a location, driven by RP_META.layers. */
  RP.nearby = (lat, lon) => {
    const layers = RP_META.layers.map((layer) => {
      if (!indexes[layer.source])
        indexes[layer.source] = new GridIndex(RP_POI[layer.source].map((p) => ({ lat: p[0] / E5, lon: p[1] / E5, name: p[2] })));
      const idx = indexes[layer.source];
      return { layer, nearest: idx.nearest(lat, lon), count: idx.countWithin(lat, lon, layer.radius) };
    });
    return { layers, distCenter: RP.haversine(lat, lon, RP_META.center.lat, RP_META.center.lon) };
  };

  // ---------------------------------------------------------------- parcels (gush / chelka -> coordinates)
  const parcels = {}, loadedChunks = new Set();
  RP.parcelChunk = (key, data) => { Object.assign(parcels, data); loadedChunks.add(key); }; // called by data/parcels_*.js

  RP.loadScript = (src) => new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = src; s.onload = resolve; s.onerror = () => reject(new Error("load failed: " + src));
    document.head.appendChild(s);
  });

  /** Binary search over the flat [chelka, lat, lon, ...] array of a gush (sorted by chelka). */
  RP.findParcel = async (gush, chelka) => {
    const chunk = Math.floor(gush / RP_META.parcelChunk);
    if (!loadedChunks.has(chunk)) await RP.loadScript(`data/parcels_${chunk}.js`).catch(() => null);
    const arr = parcels[gush];
    if (!arr) return null;
    let lo = 0, hi = arr.length / 3 - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1, c = arr[mid * 3];
      if (c === chelka) return { lat: arr[mid * 3 + 1] / E5, lon: arr[mid * 3 + 2] / E5, exact: true };
      c < chelka ? (lo = mid + 1) : (hi = mid - 1);
    }
    let la = 0, lo2 = 0; const n = arr.length / 3; // chelka unknown: use the gush centre
    for (let i = 0; i < n; i++) { la += arr[i * 3 + 1]; lo2 += arr[i * 3 + 2]; }
    return { lat: la / n / E5, lon: lo2 / n / E5, exact: false };
  };

  // ---------------------------------------------------------------- address search (OpenStreetMap Nominatim)
  RP.geocode = async (query) => {
    const url = "https://nominatim.openstreetmap.org/search?format=jsonv2&limit=1&countrycodes=il&accept-language=he&q=" + encodeURIComponent(query);
    const res = await fetch(url);
    if (!res.ok) throw new Error("geocoder unavailable");
    const hit = (await res.json())[0];
    return hit ? { lat: +hit.lat, lon: +hit.lon, label: hit.display_name } : null;
  };

  // ---------------------------------------------------------------- map (free OpenStreetMap, no API key required)
  RP.icon = (emoji, cls) => L.divIcon({ html: `<span class="pin ${cls}">${emoji}</span>`, className: "", iconSize: [34, 34], iconAnchor: [17, 17] });

  RP.createMap = (el, onClick) => {
    const map = L.map(el).setView([31.6, 34.9], 8);
    L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}", {
      maxZoom: 19,
      attribution: "Tiles &copy; Esri"
    }).addTo(map);
    if (onClick) map.on("click", (e) => onClick({ lat: e.latlng.lat, lon: e.latlng.lng, label: "נקודה שנבחרה במפה" }));
    return map;
  };

  /** Plot the selected property along with the nearest school and transit stop on the map. */
  RP.showLocationOnMap = (map, layerGroup, loc) => {
    layerGroup.clearLayers();
    L.marker([loc.lat, loc.lon], { icon: RP.icon("📍", "home") })
      .addTo(layerGroup)
      .bindPopup(loc.label)
      .openPopup();

    const near = RP.nearby(loc.lat, loc.lon);
    const bounds = [[loc.lat, loc.lon]];

    near.layers.forEach(({ layer: meta, nearest }) => {
      if (!nearest) return;
      const { p, d } = nearest;
      L.marker([p.lat, p.lon], { icon: RP.icon(meta.icon, meta.source) })
        .addTo(layerGroup)
        .bindPopup(`${meta.label}: ${p.name || "ללא שם"} (${RP.meters(d)})`);
      L.polyline([[loc.lat, loc.lon], [p.lat, p.lon]], {
        color: "#e0245e",
        weight: 2,
        dashArray: "6 6"
      }).addTo(layerGroup);
      bounds.push([p.lat, p.lon]);
    });

    map.fitBounds(bounds, { padding: [50, 50], maxZoom: 17 });
    return near;
  };

  /** Show a warning banner when the pipeline has not generated the data files yet. */
  RP.requireData = (...globals) => {
    const missing = globals.filter((g) => !window[g]);
    if (!missing.length) return true;
    const div = document.createElement("div");
    div.className = "banner";
    div.textContent = "קבצי הנתונים של האתר חסרים (" + missing.join(", ") + "). הריצו את הפייפליין: python main.py";
    document.body.prepend(div);
    return false;
  };
})();