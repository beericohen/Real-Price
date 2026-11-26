/* RealPrice page logic (Auto-Price layout): stepper form, estimate + range, deal score, surroundings, market position,
   price curves, nearby alternatives, "why this price" receipt, what-if sliders, dataset + model stats. */
(function () {
  if (!RP.requireData("RP_META", "RP_POI", "RP_MODEL")) return;
  const $ = (s) => document.querySelector(s), $$ = (s) => [...document.querySelectorAll(s)];
  const M = RP_MODEL, META = RP_META, money = RP.money, whole = RP.fmt;
  const ltr = (s) => `<bdi dir="ltr">${s}</bdi>`;
  const signed = (n) => `${n > 0 ? "+" : n < 0 ? "\u2212" : "±"}${money(Math.abs(n))}`;
  const round5k = (n) => Math.round(n / 5000) * 5000;
  const natureLabel = (c) => (c === M.other_label ? "אחר" : c);
  const thisYear = new Date().getFullYear();
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

  let loc = null, mode = "buy", last = null, wiYearTouched = false;

  // ---------------------------------------------------------------- small UI helpers
  function paintSlider(input) {
    const paint = () => input.style.setProperty("--fill", ((input.value - input.min) / (input.max - input.min)) * 100 + "%");
    input.addEventListener("input", paint); paint();
  }
  function animateNumber(el, to, duration = 650) {
    const from = Number(el.dataset.value || 0), start = performance.now();
    const step = (t) => {
      const p = Math.min(1, (t - start) / duration), e = 1 - Math.pow(1 - p, 3);
      el.textContent = whole(from + (to - from) * e);
      if (p < 1) requestAnimationFrame(step); else el.dataset.value = to;
    };
    requestAnimationFrame(step);
  }
  const restart = (el, cls) => { el.classList.remove(cls); void el.offsetWidth; el.classList.add(cls); };

  $("#theme-toggle").onclick = () => {
    const root = document.documentElement, next = root.getAttribute("data-theme") === "light" ? "dark" : "light";
    root.setAttribute("data-theme", next);
    try { localStorage.setItem("realprice-theme", next); } catch (e) {}
  };

  // ---------------------------------------------------------------- map + picker (step 1)
  const map = RP.createMap($("#map"), setLocation);
  const layer = L.layerGroup().addTo(map);
  const picker = RP.createPicker($("#picker"), setLocation);
  function setLocation(l) {
    loc = l;
    RP.showLocationOnMap(map, layer, l);
    picker.say("מיקום נבחר: " + l.label);
    $("#load-status").textContent = "";
  }

  // ---------------------------------------------------------------- form fields
  const f = { area: $("#f-area"), rooms: $("#f-rooms"), year: $("#f-year"), unknown: $("#f-year-unknown"), nature: $("#f-nature"), date: $("#f-date"), asking: $("#f-asking") };
  f.nature.innerHTML = [...M.nature_categories, M.other_label].map((c) => `<option value="${c}">${natureLabel(c)}</option>`).join("");
  f.date.value = new Date().toISOString().slice(0, 10);
  f.year.max = thisYear + 3;
  const fmts = {
    "f-area": (v) => `${v} מ״ר`, "f-rooms": (v) => v, "f-year": (v) => (f.unknown.checked ? "לא ידועה" : v),
  };
  [f.area, f.rooms, f.year].forEach((i) => {
    const out = document.querySelector(`[data-out-for="${i.id}"]`), render = () => (out.textContent = fmts[i.id](i.value));
    paintSlider(i); i.addEventListener("input", render); render();
    f.unknown.addEventListener("change", render);
  });
  f.unknown.addEventListener("change", () => { f.year.disabled = f.unknown.checked; });

  function inputs() {
    return {
      lat: loc.lat, lon: loc.lon, area: +f.area.value, rooms: +f.rooms.value,
      yearBuilt: f.unknown.checked ? null : +f.year.value, nature: f.nature.value, date: new Date(f.date.value || Date.now()),
    };
  }

  // ---------------------------------------------------------------- stepper
  const steps = $$(".form-step"), dots = $$(".step-dot"), panel = $("#valuation-form"), status = $("#load-status");
  let current = 1;
  function render(dir) {
    steps.forEach((s) => { const n = +s.dataset.step; s.classList.toggle("is-active", n === current); s.classList.toggle("dir-back", n === current && dir === "back"); });
    dots.forEach((d) => { const n = +d.dataset.dot; d.classList.toggle("active", n === current); d.classList.toggle("done", n < current); d.setAttribute("aria-selected", n === current); });
    $("#step-bar-fill").style.width = (current / steps.length) * 100 + "%";
    if (current === 1) setTimeout(() => map.invalidateSize(), 60);
  }
  function goTo(n, dir) {
    if (n > current && current === 1 && !loc) {
      status.textContent = "בחרו קודם מיקום (גוש/חלקה, כתובת או לחיצה על המפה).";
      panel.classList.remove("shake"); void panel.offsetWidth; panel.classList.add("shake"); return;
    }
    status.textContent = ""; current = n; render(dir);
  }
  $$(".step-next").forEach((b) => (b.onclick = () => goTo(+b.dataset.goto, "next")));
  $$(".step-back").forEach((b) => (b.onclick = () => goTo(+b.dataset.goto, "back")));
  dots.forEach((d) => {
    d.onclick = () => goTo(+d.dataset.dot, +d.dataset.dot < current ? "back" : "next");
    d.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); d.onclick(); } };
  });
  render();

  $$(".mode-toggle button").forEach((b) => (b.onclick = () => {
    $$(".mode-toggle button").forEach((x) => x.classList.remove("active")); b.classList.add("active");
    mode = b.dataset.mode;
    $("#submit-label").textContent = mode === "buy" ? "חשבו שווי שוק" : "חשבו מחיר מבוקש";
  }));

  // ---------------------------------------------------------------- derived views
  /** Deal score: 100 = very cheap, 0 = very expensive (same thresholds as Auto-Price). */
  function dealScore(value, asking) {
    const ratio = asking / value, score = clamp(Math.round(100 - (ratio - 0.7) * 100), 0, 100);
    if (ratio <= 0.9) return { score, label: "עסקה מצוינת", cls: "good" };
    if (ratio <= 1.05) return { score, label: "מחיר הוגן", cls: "fair" };
    return { score, label: "ייתכן שהמחיר גבוה", cls: "bad" };
  }

  /** Reliability heuristic: narrow range + exact parcel => higher. Not a statistical confidence interval. */
  function reliability(r, l, xs) {
    const width = (r.high - r.low) / r.price;
    let s = 100 - width * 60 - (l.exact === false ? 20 : 0) - (xs.yearBuilt == null ? 5 : 0);
    s = clamp(Math.round(s), 15, 95);
    return { s, level: s >= 70 ? "גבוהה" : s >= 45 ? "בינונית" : "נמוכה", width };
  }

  /** Same property at points on rings around the location. Points far from any school/stop are skipped (sea, fields). */
  function sweep(xs, rings, dirs) {
    const out = [];
    for (const d of rings) for (let k = 0; k < dirs; k++) {
      const a = (k / dirs) * 2 * Math.PI, lat = xs.lat + (d * Math.cos(a)) / 111320, lon = xs.lon + (d * Math.sin(a)) / (111320 * Math.cos((xs.lat * Math.PI) / 180));
      if (lat < META.bbox.lat[0] || lat > META.bbox.lat[1] || lon < META.bbox.lon[0] || lon > META.bbox.lon[1]) continue;
      const e = RP.estimate({ ...xs, lat, lon });
      const nearest = Math.min(...e.near.layers.map((x) => (x.nearest ? x.nearest.d : Infinity)));
      if (nearest <= 700) out.push({ ...e, lat, lon, d, k, dirs });
    }
    return out;
  }
  const DIRS8 = ["צפון", "צפון-מזרח", "מזרח", "דרום-מזרח", "דרום", "דרום-מערב", "מערב", "צפון-מערב"];

  function curves(xs, r) {
    const pt = (x, patch) => { const p = RP.predict({ ...xs, ...patch }); return { x, p: p.price, lo: p.low, hi: p.high }; };
    const area = [], year = [];
    for (let a = 20; a <= 250; a += 10) area.push(pt(a, { area: a, rooms: clamp(xs.rooms, 1, Math.max(1, a / 12)) }));
    for (let y = 1950; y <= thisYear + 2; y += 4) year.push(pt(y, { yearBuilt: y }));
    return {
      area: { pts: area, you: { x: clamp(xs.area, 20, 250), p: r.price } },
      year: { pts: year, you: xs.yearBuilt == null ? null : { x: clamp(xs.yearBuilt, 1950, thisYear + 2), p: r.price } },
    };
  }

  const chart = new CurveChart($("#curve-svg"));
  $$(".axis-toggle[aria-label='ציר'] button").forEach((b) => (b.onclick = () => {
    $$(".axis-toggle[aria-label='ציר'] button").forEach((x) => x.classList.remove("active")); b.classList.add("active"); chart.setAxis(b.dataset.axis);
  }));

  // ---------------------------------------------------------------- renderers
  function renderSurroundings(r) {
    const tiles = r.near.layers.map(({ layer, nearest, count }) =>
      `<div><strong>${nearest ? RP.meters(nearest.d) : "-"}</strong><span>${layer.icon} ${layer.label} הקרוב</span><em>${nearest ? (nearest.p.name || "ללא שם") + " · " : ""}${count} ברדיוס ${RP.meters(layer.radius)}</em></div>`);
    tiles.push(`<div><strong>${RP.meters(r.near.distCenter)}</strong><span>🏙️ מרחק ממרכז תל אביב</span><em>קו אווירי</em></div>`);
    $("#near-grid").innerHTML = tiles.join("");
  }

  function renderMarket(xs, r) {
    const pts = sweep(xs, [600, 1500, 3000], 16);
    if (pts.length < 6) {
      $("#percentile").textContent = "—"; $("#position-title").textContent = "אין מספיק נקודות להשוואה";
      $("#position-text").textContent = "באזור הזה כמעט אין נקודות ידועות (בית ספר/תחנה) לשם השוואה."; $("#market-summary").textContent = "ההשוואה לא זמינה במיקום הזה.";
    } else {
      const pct = Math.round((pts.filter((p) => p.perSqm < r.perSqm).length / pts.length) * 100);
      $("#percentile").textContent = pct; $("#position-marker").style.setProperty("--pos", pct + "%");
      $("#position-scope").textContent = "עד 3 ק״מ";
      $("#position-title").textContent = pct >= 70 ? "יקר מרוב הנקודות בסביבה" : pct <= 30 ? "זול מרוב הנקודות בסביבה" : "באמצע טווח המחירים באזור";
      $("#position-text").textContent = `מחיר המ״ר כאן גבוה מ-${pct}% מתוך ${pts.length} נקודות בטווח 3 ק״מ, לאותו נכס בדיוק (לפי המודל).`;
      $("#market-summary").textContent = `${pts.length} נקודות בסביבה הושוו לנכס שלכם, כולן מחושבות במודל.`;
    }
    chart.setData(curves(xs, r));
  }

  function renderAround(xs, r) {
    const grid = $("#similar-grid");
    const pts = sweep(xs, [1500], 8);
    if (!pts.length) { grid.innerHTML = '<div class="empty-card">לא נמצאו נקודות מאוישות בסביבה להשוואה.</div>'; return; }
    grid.innerHTML = pts.map((p) => {
      const d = p.price - r.price, near = p.near.layers.map((x) => x.nearest && `${x.layer.icon} ${RP.meters(x.nearest.d)}`).filter(Boolean).join(" · ");
      return `<article class="similar-card"><div class="similar-title"><strong>${DIRS8[p.k]}</strong><span>${RP.meters(p.d)}</span></div><div class="sub">${near || "—"}</div>
        <div class="similar-bottom"><strong>${ltr(money(round5k(p.price)))}</strong><span class="${d >= 0 ? "above" : "below"}">${ltr(signed(round5k(d)))} מההערכה</span></div></article>`;
    }).join("");
  }

  function renderDeal(price, asking) {
    const dc = $("#deal-card");
    if (!(asking > 0)) { dc.hidden = true; return; }
    const d = dealScore(price, asking), gap = Math.round((Math.abs(price - asking) / price) * 100);
    dc.hidden = false; dc.className = "deal-card " + d.cls;
    $("#deal-score").textContent = d.score + " / 100"; $("#deal-label").textContent = d.label; $("#deal-fill").style.width = d.score + "%";
    $("#deal-explain").innerHTML = `${mode === "sell" ? "המחיר שביקשתם" : "המוכר מבקש"} ${ltr(money(asking))} – ${gap}% ${asking < price ? "מתחת" : "מעל"} להערכת המודל.`;
  }

  /** "Why this price?": start from a typical flat in central Tel Aviv, swap one factor at a time (cumulative), re-run the model. */
  function renderReceipt(xs, r) {
    const C = META.center, base = { lat: C.lat, lon: C.lon, area: 90, rooms: 4, yearBuilt: null, nature: "דירה בבית קומות", date: xs.date };
    const short = (t) => (t.length > 30 ? t.slice(0, 29) + "…" : t);
    const plan = [
      ["מיקום", "מרכז תל אביב", short(loc.label), { lat: xs.lat, lon: xs.lon }],
      ["שטח", "90 מ״ר", xs.area + " מ״ר", { area: xs.area }],
      ["חדרים", "4", String(xs.rooms), { rooms: xs.rooms }],
      ["שנת בנייה", "לא ידועה", xs.yearBuilt ?? "לא ידועה", { yearBuilt: xs.yearBuilt }],
      ["סוג נכס", "דירה בבית קומות", natureLabel(xs.nature), { nature: xs.nature }],
    ];
    let cur = { ...base }, prev = RP.estimate(cur).price; const startPrice = prev, rows = [];
    for (const [label, from, to, patch] of plan) {
      const next = { ...cur, ...patch }, p = RP.estimate(next).price;
      if (String(from) !== String(to) || Math.abs(p - prev) >= 1) rows.push({ label, from, to, delta: p - prev });
      cur = next; prev = p;
    }
    $("#receipt-empty").hidden = true; $("#receipt").hidden = false;
    $("#receipt-car").textContent = `${xs.area} מ״ר · ${xs.rooms} חד׳`;
    $("#receipt-date").textContent = new Date().toLocaleDateString("he-IL", { year: "numeric", month: "short", day: "numeric" });
    $("#receipt-base-value").textContent = money(startPrice);
    $("#receipt-base-note").textContent = "(דירה טיפוסית: 90 מ״ר, 4 חדרים, מרכז תל אביב)";
    $("#receipt-items").innerHTML = rows.map((it, i) => `<div class="receipt-item" style="animation-delay:${i * 90}ms"><div class="ri-text"><span class="ri-label">${it.label}</span><span class="ri-detail">${it.from} ← ${it.to}</span></div><span class="ri-amount ${it.delta >= 0 ? "pos" : "neg"}">${signed(it.delta)}</span></div>`).join("");
    $("#receipt-total").textContent = money(r.price);
    $("#receipt-note").textContent = "הערכת הבסיס + כל השורות מעלה משחזרים בדיוק את חיזוי המודל (לפני עיגול).";
  }

  function renderNatureBars(xs) {
    const rows = [...M.nature_categories, M.other_label].map((n) => [natureLabel(n), RP.estimate({ ...xs, nature: n }).price]);
    const max = Math.max(...rows.map((r) => r[1])) || 1;
    $("#nature-bars").innerHTML = rows.map(([k, v]) => `<div class="bar-row wide"><span>${k}</span><div><i style="width:${(v / max) * 100}%"></i></div><b>${money(round5k(v))}</b></div>`).join("") + '<p class="note">אותו שטח, חדרים ומיקום – רק סוג הנכס משתנה.</p>';
  }

  // ---------------------------------------------------------------- submit
  panel.onsubmit = async (e) => {
    e.preventDefault();
    if (!loc) { goTo(1, "back"); status.textContent = "בחרו קודם מיקום."; return; }
    status.textContent = "מנתח נתוני שוק…"; panel.classList.add("is-loading"); $(".calc-submit").classList.add("is-busy");
    await new Promise((r) => setTimeout(r, 120));
    try {
      const xs = inputs(); RP.showLocationOnMap(map, layer, loc);
      const r = RP.predict(xs); last = { xs, r };
      animateNumber($("#price-value"), round5k(r.price)); restart($("#price-value"), "bump");
      $("#range-low").textContent = money(round5k(r.low)); $("#range-high").textContent = money(round5k(r.high));
      $("#range-note").textContent = `טווח סביר (${Math.round((M.interval[1] - M.interval[0]) * 100)}%) – אומדן סטטיסטי, לא שמאות.`;

      const rel = reliability(r, loc, xs);
      $("#coverage-level").textContent = rel.level; $("#coverage-fill").style.width = rel.s + "%";
      $("#coverage-text").textContent = `רוחב הטווח הוא ${Math.round(rel.width * 100)}% מהמחיר${loc.exact === false ? "; המיקום משוער (מרכז הגוש)" : ""}. מדד יוריסטי, לא רווח סמך סטטיסטי.`;
      renderDeal(r.price, +f.asking.value);
      renderSurroundings(r); renderMarket(xs, r); renderAround(xs, r); renderReceipt(xs, r); renderNatureBars(xs);

      // seed what-if from this result
      $("#whatif-area").value = xs.area; $("#whatif-rooms").value = xs.rooms;
      wiYearTouched = xs.yearBuilt != null; $("#whatif-year").value = xs.yearBuilt ?? 1995;
      $$("#whatif-area,#whatif-rooms,#whatif-year").forEach((el) => el.dispatchEvent(new Event("input")));
      $("#whatif-current").textContent = money(round5k(r.price)); $("#whatif-new").textContent = money(round5k(r.price));
      $("#whatif-delta").textContent = "הזיזו סליידר כדי להריץ חיזוי נוסף של המודל.";
      status.textContent = "ההערכה הושלמה";
      if (window.innerWidth < 900) $(".result-panel").scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      console.error(err); status.textContent = "לא הצלחנו לחשב הערכה. בדקו את הפרטים ונסו שוב.";
    } finally { panel.classList.remove("is-loading"); $(".calc-submit").classList.remove("is-busy"); }
  };

  // ---------------------------------------------------------------- what-if
  const wi = { area: $("#whatif-area"), rooms: $("#whatif-rooms"), year: $("#whatif-year") };
  const wiOut = { area: $("#whatif-area-value"), rooms: $("#whatif-rooms-value"), year: $("#whatif-year-value") };
  wi.year.max = thisYear + 3;
  const wiLabel = { area: (v) => `${v} מ״ר`, rooms: (v) => v, year: (v) => (wiYearTouched ? v : "לא ידועה") };
  Object.keys(wi).forEach((k) => {
    paintSlider(wi[k]); wiOut[k].textContent = wiLabel[k](wi[k].value);
    wi[k].addEventListener("input", (e) => {
      if (k === "year" && e.isTrusted) wiYearTouched = true;
      wiOut[k].textContent = wiLabel[k](wi[k].value);
      if (!last || !e.isTrusted) return;
      const xs = { ...last.xs, area: +wi.area.value, rooms: +wi.rooms.value, yearBuilt: wiYearTouched ? +wi.year.value : last.xs.yearBuilt };
      const p = RP.estimate(xs).price, d = p - last.r.price;
      $("#whatif-new").textContent = money(round5k(p)); restart($("#whatif-new"), "pulse");
      const note = $("#whatif-delta"); note.innerHTML = `${ltr(signed(round5k(d)))} לעומת ההערכה הנוכחית · הרצה אמיתית של המודל`; restart(note, "flash");
    });
  });

  // ---------------------------------------------------------------- dataset + model stats
  const m = META.metrics, pct = (v, d = 0) => (v * 100).toFixed(d) + "%";
  $("#dash-deals").textContent = whole(META.nDeals); $("#dash-parcels").textContent = whole(META.nParcels);
  $("#dash-schools").textContent = whole(RP_POI.schools.length); $("#dash-transit").textContent = whole(RP_POI.transit.length);
  $("#dash-r2").textContent = m.r2.toFixed(3); $("#dash-ape").textContent = pct(m.median_ape, 1);
  $("#accuracy-bars").innerHTML = [["בטווח 10% מהמחיר", m.within_10pct], ["בטווח 20% מהמחיר", m.within_20pct], ["כיסוי הטווח המוצג", M.interval_coverage]]
    .map(([k, v]) => `<div class="bar-row wide"><span>${k}</span><div><i style="width:${v * 100}%"></i></div><b>${pct(v)}</b></div>`).join("");

  $("#stat-r2").textContent = m.r2.toFixed(4); $("#stat-mae").textContent = money(m.mae_nis); $("#stat-rmse").textContent = money(m.rmse_nis);
  $("#stat-rows").textContent = whole(M.n_train);
  $("#stat-trained").textContent = new Date(M.trained_at + "T00:00:00").toLocaleDateString("he-IL", { year: "numeric", month: "short", day: "numeric" });
  $("#stat-features").textContent = M.feature_names.length; $("#stat-trees").textContent = whole(M.models.median.roots.length);
  $("#stat-lo").textContent = Math.round(M.interval[0] * 100); $("#stat-hi").textContent = Math.round(M.interval[1] * 100);

  /** Importance = share of splits that use each factor group in the median model (split frequency, not gain). */
  (function importance() {
    const med = M.models.median, count = new Array(M.feature_names.length).fill(0);
    for (let i = 0; i < med.f.length; i++) if (med.f[i] >= 0) count[med.f[i]]++;
    const groupOf = (n) => /^(lat|long|distance_from_center)$/.test(n) ? "מיקום" : /^(asset_area|room_num|area_per_room)$/.test(n) ? "שטח וחדרים"
      : /^(year_built|age)$/.test(n) ? "גיל הנכס" : /^deal_/.test(n) ? "תאריך העסקה" : /school|transit|public_transport/.test(n) ? "קרבה לשירותים" : "סוג נכס";
    const g = {}; M.feature_names.forEach((n, i) => (g[groupOf(n)] = (g[groupOf(n)] || 0) + count[i]));
    const total = Object.values(g).reduce((a, b) => a + b, 0) || 1;
    $("#importance-method").textContent = "תדירות השימוש בכל קבוצת גורמים בפיצולי העצים של מודל החציון – מדד גס, לא תרומה למחיר.";
    $("#importance-bars").innerHTML = Object.entries(g).sort((a, b) => b[1] - a[1]).map(([k, v]) => {
      const p = Math.round((v / total) * 100);
      return `<div class="imp-row"><span>${k}</span><div class="imp-track"><i class="imp-fill" style="--pct:${p}%"></i></div><b>${p}%</b></div>`;
    }).join("");
  })();

  $("#live-dot").textContent = "המודל מוכן";
  $("#cta-start").onclick = () => $("#valuation").scrollIntoView({ behavior: "smooth" });
  const io = new IntersectionObserver((es) => es.forEach((e) => e.isIntersecting && e.target.classList.add("in-view")), { threshold: 0.08 });
  $$(".section,.hero-copy,.hero-visual").forEach((e) => io.observe(e));
})();
