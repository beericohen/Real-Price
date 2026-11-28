/* Shared helpers: number formatting, distance maths, a spatial index for schools / bus stops,
   lazy loading of parcel coordinates, address search and the map. Everything is attached to one global object, RP. */
(function () {
  const RP = (window.RP = window.RP || {});

  // Coordinates in the data files are whole numbers: degrees * 100,000 (about 1 metre precision, small files).
  const COORD_SCALE = 1e5;

  // ---------------------------------------------------------------- formatting
  const numberFormat = new Intl.NumberFormat("he-IL");
  RP.fmt = (n) => numberFormat.format(Math.round(n));
  RP.money = (n) => "₪" + RP.fmt(n);
  RP.meters = (m) => (m >= 1000 ? (m / 1000).toFixed(2) + " ק״מ" : RP.fmt(m) + " מ׳");

  // The deal type that the model groups all rare types into is shown as "other"
  RP.natureLabel = (nature) => (nature === RP_MODEL.other_label ? "אחר" : nature);

  // ---------------------------------------------------------------- distance
  /** Distance in metres between two GPS points (Haversine formula). */
  RP.haversine = (lat1, lon1, lat2, lon2) => {
    const toRad = Math.PI / 180;
    const dLat = (lat2 - lat1) * toRad;
    const dLon = (lon2 - lon1) * toRad;
    const a = Math.sin(dLat / 2) ** 2 + Math.cos(lat1 * toRad) * Math.cos(lat2 * toRad) * Math.sin(dLon / 2) ** 2;
    return 12742000 * Math.asin(Math.sqrt(a)); // 12,742 km = the earth's diameter
  };

  // ---------------------------------------------------------------- spatial index
  /**
   * Splits the map into square cells and remembers which points are in which cell, so that a search only has to
   * look at the cells around the query point instead of at all ~20,000 points.
   * (The same idea as the KD-tree that the Python pipeline uses for the same two questions.)
   */
  class GridIndex {
    constructor(points, cellSize = 0.01) {
      this.points = points; // [{lat, lon, name}]
      this.cellSize = cellSize; // in degrees (0.01 is about 1 km)
      this.cells = new Map(); // "i,j" -> indexes of the points inside that cell
      // Width of a cell in metres at the northern end of Israel, where cells are narrowest. A point further than
      // r cells away is therefore at least r * minCellMeters away - this is what makes the search below correct.
      this.minCellMeters = cellSize * 111320 * Math.cos((33.4 * Math.PI) / 180);

      points.forEach((point, index) => {
        const key = this.cellKey(Math.floor(point.lat / cellSize), Math.floor(point.lon / cellSize));
        if (!this.cells.has(key)) this.cells.set(key, []);
        this.cells.get(key).push(index);
      });
    }

    cellKey(i, j) {
      return i + "," + j;
    }

    /** All points in the cells that are exactly `ring` cells away from cell (ci, cj). Ring 0 is the cell itself. */
    pointsInRing(ci, cj, ring) {
      const found = [];
      for (let di = -ring; di <= ring; di++) {
        for (let dj = -ring; dj <= ring; dj++) {
          if (Math.max(Math.abs(di), Math.abs(dj)) !== ring) continue; // inner cells belong to smaller rings
          const cell = this.cells.get(this.cellKey(ci + di, cj + dj)) || [];
          for (const index of cell) found.push(this.points[index]);
        }
      }
      return found;
    }

    /** The nearest point as {d (metres), p (the point)}, or null. Searches ring after ring around the query. */
    nearest(lat, lon, maxRings = 60) {
      const ci = Math.floor(lat / this.cellSize);
      const cj = Math.floor(lon / this.cellSize);
      let best = null;
      for (let ring = 0; ring <= maxRings; ring++) {
        for (const p of this.pointsInRing(ci, cj, ring)) {
          const d = RP.haversine(lat, lon, p.lat, p.lon);
          if (!best || d < best.d) best = { d, p };
        }
        // Everything in the next ring is further away than this, so a closer point is already a sure winner
        if (best && best.d <= ring * this.minCellMeters) break;
      }
      return best;
    }

    /** How many points are within `radius` metres. */
    countWithin(lat, lon, radius) {
      const ci = Math.floor(lat / this.cellSize);
      const cj = Math.floor(lon / this.cellSize);
      const rings = Math.ceil(radius / this.minCellMeters);
      let count = 0;
      for (let ring = 0; ring <= rings; ring++) {
        for (const p of this.pointsInRing(ci, cj, ring)) {
          if (RP.haversine(lat, lon, p.lat, p.lon) <= radius) count++;
        }
      }
      return count;
    }
  }

  // ---------------------------------------------------------------- points of interest (schools, bus stops)
  const poiIndexes = {}; // one index per layer, built the first time it is needed

  /** Nearest point and number of points within the layer's radius, for every layer in RP_META.layers. */
  RP.nearby = (lat, lon) => {
    const layers = RP_META.layers.map((layer) => {
      if (!poiIndexes[layer.source]) {
        const points = RP_POI[layer.source].map((p) => ({ lat: p[0] / COORD_SCALE, lon: p[1] / COORD_SCALE, name: p[2] }));
        poiIndexes[layer.source] = new GridIndex(points);
      }
      const index = poiIndexes[layer.source];
      return { layer, nearest: index.nearest(lat, lon), count: index.countWithin(lat, lon, layer.radius) };
    });
    return { layers, distCenter: RP.haversine(lat, lon, RP_META.center.lat, RP_META.center.lon) };
  };

  // ---------------------------------------------------------------- parcels: gush + chelka -> coordinates
  const parcels = {}; // gush -> flat array [chelka, lat, lon, chelka, lat, lon, ...] sorted by chelka
  const loadedChunks = new Set();

  /** Called by the data files (data/parcels_<n>.js) when they are loaded. */
  RP.parcelChunk = (chunk, data) => {
    Object.assign(parcels, data);
    loadedChunks.add(chunk);
  };

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = src;
      script.onload = resolve;
      script.onerror = () => reject(new Error("load failed: " + src));
      document.head.appendChild(script);
    });
  }

  /**
   * Coordinates of a parcel: {lat, lon, exact}, or null if the gush is unknown.
   * If the chelka is not in the data, the centre of the whole gush is returned with exact = false.
   */
  RP.findParcel = async (gush, chelka) => {
    // Parcels are split into one file per 100 gushim and loaded only when needed
    const chunk = Math.floor(gush / RP_META.parcelChunk);
    if (!loadedChunks.has(chunk)) await loadScript(`data/parcels_${chunk}.js`).catch(() => null);

    const flat = parcels[gush];
    if (!flat) return null;
    const count = flat.length / 3;

    // Binary search (the chelkot are sorted)
    let low = 0;
    let high = count - 1;
    while (low <= high) {
      const mid = (low + high) >> 1;
      const midChelka = flat[mid * 3];
      if (midChelka === chelka) {
        return { lat: flat[mid * 3 + 1] / COORD_SCALE, lon: flat[mid * 3 + 2] / COORD_SCALE, exact: true };
      }
      if (midChelka < chelka) low = mid + 1;
      else high = mid - 1;
    }

    // Chelka not found: use the average position of the gush
    let latSum = 0;
    let lonSum = 0;
    for (let i = 0; i < count; i++) {
      latSum += flat[i * 3 + 1];
      lonSum += flat[i * 3 + 2];
    }
    return { lat: latSum / count / COORD_SCALE, lon: lonSum / count / COORD_SCALE, exact: false };
  };

  // ---------------------------------------------------------------- address search (OpenStreetMap Nominatim, free, no API key)
  RP.geocode = async (query) => {
    const url =
      "https://nominatim.openstreetmap.org/search?format=jsonv2&limit=1&countrycodes=il&accept-language=he&q=" +
      encodeURIComponent(query);
    const response = await fetch(url);
    if (!response.ok) throw new Error("geocoder unavailable");
    const hit = (await response.json())[0];
    return hit ? { lat: +hit.lat, lon: +hit.lon, label: hit.display_name } : null;
  };

  // ---------------------------------------------------------------- map (Leaflet + free map tiles, no API key)
  function pinIcon(emoji, cssClass) {
    return L.divIcon({
      html: `<span class="pin ${cssClass}">${emoji}</span>`,
      className: "",
      iconSize: [34, 34],
      iconAnchor: [17, 17],
    });
  }

  /** Creates the map. `onClick` receives {lat, lon, label} when the user clicks a point. */
  RP.createMap = (element, onClick) => {
    const map = L.map(element).setView([31.6, 34.9], 8); // the whole of Israel
    L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}", {
      maxZoom: 19,
      attribution: "Tiles &copy; Esri",
    }).addTo(map);
    if (onClick) {
      map.on("click", (e) => onClick({ lat: e.latlng.lat, lon: e.latlng.lng, label: "נקודה שנבחרה במפה" }));
    }
    return map;
  };

  /** Shows the property and its nearest school and bus stop on the map. Returns the nearby-places result. */
  RP.showLocationOnMap = (map, layerGroup, loc) => {
    layerGroup.clearLayers();
    L.marker([loc.lat, loc.lon], { icon: pinIcon("📍", "home") })
      .addTo(layerGroup)
      .bindPopup(loc.label)
      .openPopup();

    const near = RP.nearby(loc.lat, loc.lon);
    const bounds = [[loc.lat, loc.lon]];

    near.layers.forEach(({ layer, nearest }) => {
      if (!nearest) return;
      const { p, d } = nearest;
      L.marker([p.lat, p.lon], { icon: pinIcon(layer.icon, layer.source) })
        .addTo(layerGroup)
        .bindPopup(`${layer.label}: ${p.name || "ללא שם"} (${RP.meters(d)})`);
      L.polyline([[loc.lat, loc.lon], [p.lat, p.lon]], { color: "#e0245e", weight: 2, dashArray: "6 6" }).addTo(layerGroup);
      bounds.push([p.lat, p.lon]);
    });

    map.fitBounds(bounds, { padding: [50, 50], maxZoom: 17 });
    return near;
  };

  /** Shows a warning at the top of the page if the data files from the pipeline are missing. */
  RP.requireData = (...names) => {
    const missing = names.filter((name) => !window[name]);
    if (!missing.length) return true;
    const banner = document.createElement("div");
    banner.className = "banner";
    banner.textContent = "קבצי הנתונים של האתר חסרים (" + missing.join(", ") + "). הריצו את הפייפליין: python main.py";
    document.body.prepend(banner);
    return false;
  };
})();
