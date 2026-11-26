/* Price model evaluated in the browser. Feature formulas mirror realprice/preprocessing.py exactly. */
(function () {
  /** Sum of leaf values over all boosted trees (flat-array layout written by realprice/exporting.py). */
  RP.score = (m, x) => {
    let total = m.base;
    for (const root of m.roots) {
      let i = root;
      while (m.f[i] >= 0) {   // x[...] is NaN when the value is unknown (e.g. no year built): follow the learned default branch
        const v = x[m.f[i]];
        i = (Number.isNaN(v) ? m.m[i] === 1 : v <= m.t[i]) ? m.l[i] : m.r[i];
      }
      total += m.t[i];
    }
    return total;
  };

  /** Model input vector in the column order stored with the model. */
  RP.buildVector = (M, inp) => {
    const near = RP.nearby(inp.lat, inp.lon);
    const d = inp.date, year = new Date().getFullYear();
    const v = {
      lat: inp.lat, long: inp.lon, year_built: inp.yearBuilt ?? NaN,          // NaN = unknown year (optional field)
      age: inp.yearBuilt == null ? NaN : Math.max(0, year - inp.yearBuilt),
      asset_area: inp.area, room_num: inp.rooms, deal_day: d.getDate(), deal_month: d.getMonth() + 1,
      deal_year: d.getFullYear(), distance_from_center: near.distCenter,
      area_per_room: inp.area / inp.rooms, deal_t: d.getFullYear() + d.getMonth() / 12,
    };
    near.layers.forEach(({ layer, nearest, count }) => {
      v[layer.dist_col] = nearest ? nearest.d : 5000; // no POI found within the search rings
      v[layer.count_col] = count;
    });
    const nature = M.nature_categories.includes(inp.nature) ? inp.nature : M.other_label;
    v[M.nature_prefix + nature] = 1;
    return { x: M.feature_names.map((n) => v[n] ?? 0), near, features: v };
  };

  /** -> {price, low, high, perSqm} (NIS). The model predicts log(price per sqm): exponentiate, multiply by the area. */
  RP.predict = (inp) => {
    const M = window.RP_MODEL, { x, near } = RP.buildVector(M, inp);
    const [low, high] = [RP.score(M.models.low, x), RP.score(M.models.high, x)].sort((a, b) => a - b);
    const perSqm = Math.exp(RP.score(M.models.median, x)), price = perSqm * inp.area;
    return { price, low: Math.min(Math.exp(low) * inp.area, price), high: Math.max(Math.exp(high) * inp.area, price), perSqm, near };
  };

  /** Fast path: median estimate only (no low/high models). Used for sweeps over many points / property variants. */
  RP.estimate = (inp) => {
    const M = window.RP_MODEL, { x, near } = RP.buildVector(M, inp);
    const perSqm = Math.exp(RP.score(M.models.median, x));
    return { price: perSqm * inp.area, perSqm, near };
  };
})();
