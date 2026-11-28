/* What the model tells us about a property, beyond the single price: deal score, reliability, the same
   property at other places, price curves, the "why this price" receipt and so on.
   These are plain functions that only calculate: nothing here touches the page (that is app.js's job).
   `inputs` always has the shape {lat, lon, area, rooms, yearBuilt (or null), nature, date}. */
(function () {
  const thisYear = new Date().getFullYear();

  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
  const round5k = (n) => Math.round(n / 5000) * 5000; // prices are shown rounded to 5,000 NIS (a model has no more precision)

  /** How good is the asking price? Score: 100 = very cheap, 0 = very expensive. */
  function dealScore(estimate, asking) {
    const ratio = asking / estimate; // 1.0 = the asking price equals the estimate
    const score = clamp(Math.round(100 - (ratio - 0.7) * 100), 0, 100);
    if (ratio <= 0.9) return { score, label: "עסקה מצוינת", cls: "good" };
    if (ratio <= 1.05) return { score, label: "מחיר הוגן", cls: "fair" };
    return { score, label: "ייתכן שהמחיר גבוה", cls: "bad" };
  }

  /**
   * A simple 15-95 reliability score. It is a rule of thumb, NOT a statistical confidence interval:
   * a narrow price range is better, and so is an exact parcel and a known year of construction.
   */
  function reliability(result, loc, inputs) {
    const width = (result.high - result.low) / result.price; // width of the range as a share of the price
    let score = 100 - width * 60;
    if (loc.exact === false) score -= 20; // only the centre of the gush is known
    if (inputs.yearBuilt == null) score -= 5;
    score = clamp(Math.round(score), 15, 95);
    const level = score >= 70 ? "גבוהה" : score >= 45 ? "בינונית" : "נמוכה";
    return { score, level, width };
  }

  /**
   * The same property at points on circles around its location: for each distance in `distances` (metres),
   * `directions` points spread evenly (direction k = 0 is north, then clockwise).
   * Points outside Israel, or far from any school or stop (sea, fields), are skipped.
   */
  function sweep(inputs, distances, directions) {
    const bbox = RP_META.bbox;
    const points = [];
    for (const distance of distances) {
      for (let k = 0; k < directions; k++) {
        const angle = (k / directions) * 2 * Math.PI;
        // 1 degree of latitude = 111,320 m; a degree of longitude is shorter the further north we are
        const lat = inputs.lat + (distance * Math.cos(angle)) / 111320;
        const lon = inputs.lon + (distance * Math.sin(angle)) / (111320 * Math.cos((inputs.lat * Math.PI) / 180));
        if (lat < bbox.lat[0] || lat > bbox.lat[1] || lon < bbox.lon[0] || lon > bbox.lon[1]) continue;

        const estimate = RP.estimate({ ...inputs, lat, lon });
        const nearestPlace = Math.min(...estimate.near.layers.map((l) => (l.nearest ? l.nearest.d : Infinity)));
        if (nearestPlace <= 700) points.push({ ...estimate, lat, lon, d: distance, k });
      }
    }
    return points;
  }

  /** The data for the two price curves: price against area, and price against year built. */
  function priceCurves(inputs, result) {
    // One point of a curve: the model's price and range with one thing changed
    const point = (x, change) => {
      const p = RP.predict({ ...inputs, ...change });
      return { x, p: p.price, lo: p.low, hi: p.high };
    };

    const byArea = [];
    for (let area = 20; area <= 250; area += 10) {
      // keep the number of rooms realistic for the area (at least 12 sqm per room)
      byArea.push(point(area, { area, rooms: clamp(inputs.rooms, 1, Math.max(1, area / 12)) }));
    }
    const byYear = [];
    for (let year = 1950; year <= thisYear + 2; year += 4) byYear.push(point(year, { yearBuilt: year }));

    return {
      area: { pts: byArea, you: { x: clamp(inputs.area, 20, 250), p: result.price } },
      year: { pts: byYear, you: inputs.yearBuilt == null ? null : { x: clamp(inputs.yearBuilt, 1950, thisYear + 2), p: result.price } },
    };
  }

  /**
   * The "why this price?" receipt. Start from a typical flat in central Tel Aviv, then swap in the user's
   * location, area, rooms, year and type ONE AT A TIME, each time running the model again.
   * The price change caused by each swap is that factor's line on the receipt.
   * Returns {startPrice, rows: [{label, from, to, delta}]}.
   */
  function receipt(inputs, locationLabel) {
    const center = RP_META.center;
    const typical = { lat: center.lat, lon: center.lon, area: 90, rooms: 4, yearBuilt: null, nature: "דירה בבית קומות", date: inputs.date };
    const shorten = (text) => (text.length > 30 ? text.slice(0, 29) + "…" : text);

    // [label, value in the typical flat, the user's value, what to change in the inputs]
    const steps = [
      ["מיקום", "מרכז תל אביב", shorten(locationLabel), { lat: inputs.lat, lon: inputs.lon }],
      ["שטח", "90 מ״ר", inputs.area + " מ״ר", { area: inputs.area }],
      ["חדרים", "4", String(inputs.rooms), { rooms: inputs.rooms }],
      ["שנת בנייה", "לא ידועה", inputs.yearBuilt ?? "לא ידועה", { yearBuilt: inputs.yearBuilt }],
      ["סוג נכס", "דירה בבית קומות", RP.natureLabel(inputs.nature), { nature: inputs.nature }],
    ];

    let current = typical;
    let previousPrice = RP.estimate(current).price;
    const startPrice = previousPrice;
    const rows = [];
    for (const [label, from, to, change] of steps) {
      current = { ...current, ...change };
      const price = RP.estimate(current).price;
      // a factor that is the same as the typical flat and changes nothing is not worth a line
      if (String(from) !== String(to) || Math.abs(price - previousPrice) >= 1) {
        rows.push({ label, from, to, delta: price - previousPrice });
      }
      previousPrice = price;
    }
    return { startPrice, rows };
  }

  /** The estimated price of the same property for every deal type: [{label, price}]. */
  function priceByNature(inputs) {
    return [...RP_MODEL.nature_categories, RP_MODEL.other_label].map((nature) => ({
      label: RP.natureLabel(nature),
      price: RP.estimate({ ...inputs, nature }).price,
    }));
  }

  /**
   * How much each group of factors is used by the model: the share of all tree splits that test a feature of
   * that group (in the median model). A rough measure of usage, not of the factor's effect on the price.
   * Returns [{name, percent}], biggest first.
   */
  function factorImportance() {
    const median = RP_MODEL.models.median;
    const names = RP_MODEL.feature_names;

    const splitsPerFeature = new Array(names.length).fill(0);
    for (const feature of median.f) if (feature >= 0) splitsPerFeature[feature]++;

    const groupOf = (name) => {
      if (/^(lat|long|distance_from_center)$/.test(name)) return "מיקום";
      if (/^(asset_area|room_num|area_per_room)$/.test(name)) return "שטח וחדרים";
      if (/^(year_built|age)$/.test(name)) return "גיל הנכס";
      if (/^deal_/.test(name)) return "תאריך העסקה";
      if (/school|transit|public_transport/.test(name)) return "קרבה לשירותים";
      return "סוג נכס"; // what is left are the one-hot columns of the deal type
    };
    const splitsPerGroup = {};
    names.forEach((name, i) => (splitsPerGroup[groupOf(name)] = (splitsPerGroup[groupOf(name)] || 0) + splitsPerFeature[i]));

    const total = Object.values(splitsPerGroup).reduce((a, b) => a + b, 0) || 1;
    return Object.entries(splitsPerGroup)
      .sort((a, b) => b[1] - a[1])
      .map(([name, splits]) => ({ name, percent: Math.round((splits / total) * 100) }));
  }

  RP.analysis = { clamp, round5k, dealScore, reliability, sweep, priceCurves, receipt, priceByNature, factorImportance };
})();
