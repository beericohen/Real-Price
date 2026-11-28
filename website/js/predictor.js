/* Runs the price model in the browser.
   The model is a set of boosted decision trees exported by realprice/exporting.py (see its docstring for the layout).
   The features built here must match the ones the model was trained on (realprice/preprocessing.py: web_matrix). */
(function () {
  /** Prediction of one exported model: walk down every tree and add up the leaf values. */
  RP.score = (model, x) => {
    let total = model.base;
    for (const root of model.roots) {
      let node = root;
      while (model.f[node] >= 0) { // f < 0 means "leaf"
        const value = x[model.f[node]];
        // An unknown value (NaN, e.g. no year built) follows the branch that the model learned for missing values
        const goLeft = Number.isNaN(value) ? model.m[node] === 1 : value <= model.t[node];
        node = goLeft ? model.l[node] : model.r[node];
      }
      total += model.t[node]; // in a leaf, t holds the leaf's value
    }
    return total;
  };

  /**
   * The model's input for one property: {x: numbers in the order of M.feature_names, near: nearby places}.
   * inp = {lat, lon, area, rooms, yearBuilt (or null), nature, date}
   */
  RP.buildVector = (M, inp) => {
    const near = RP.nearby(inp.lat, inp.lon);
    const date = inp.date;
    const thisYear = new Date().getFullYear();

    const values = {
      lat: inp.lat,
      long: inp.lon,
      year_built: inp.yearBuilt ?? NaN, // NaN = the user does not know the year
      age: inp.yearBuilt == null ? NaN : Math.max(0, thisYear - inp.yearBuilt),
      asset_area: inp.area,
      room_num: inp.rooms,
      deal_day: date.getDate(),
      deal_month: date.getMonth() + 1,
      deal_year: date.getFullYear(),
      distance_from_center: near.distCenter,
      area_per_room: inp.area / inp.rooms,
      deal_t: date.getFullYear() + date.getMonth() / 12, // time as one continuous number
    };
    near.layers.forEach(({ layer, nearest, count }) => {
      values[layer.dist_col] = nearest ? nearest.d : 5000; // nothing found nearby: a large distance
      values[layer.count_col] = count;
    });

    // The deal type is one-hot encoded: 1 in the column of this type, and 0 (the default below) in the others
    const nature = M.nature_categories.includes(inp.nature) ? inp.nature : M.other_label;
    values[M.nature_prefix + nature] = 1;

    return { x: M.feature_names.map((name) => values[name] ?? 0), near };
  };

  /**
   * Full estimate -> {price, low, high, perSqm, near}, in NIS.
   * The models predict log(price per sqm), so: price = exp(prediction) * area.
   */
  RP.predict = (inp) => {
    const M = window.RP_MODEL;
    const { x, near } = RP.buildVector(M, inp);

    const perSqm = Math.exp(RP.score(M.models.median, x));
    const price = perSqm * inp.area;
    // The two percentile models can cross each other, so sort them, and keep the price inside the range
    const [lowLog, highLog] = [RP.score(M.models.low, x), RP.score(M.models.high, x)].sort((a, b) => a - b);
    return {
      price,
      low: Math.min(Math.exp(lowLog) * inp.area, price),
      high: Math.max(Math.exp(highLog) * inp.area, price),
      perSqm,
      near,
    };
  };

  /** Faster: the median only (one model instead of three). Used when many variations are calculated at once. */
  RP.estimate = (inp) => {
    const M = window.RP_MODEL;
    const { x, near } = RP.buildVector(M, inp);
    const perSqm = Math.exp(RP.score(M.models.median, x));
    return { price: perSqm * inp.area, perSqm, near };
  };
})();
