/* The page itself: reads the form, shows the results and handles clicks.
   All calculations are done by predictor.js and analysis.js; this file only deals with the page (the DOM). */
(function () {
  if (!RP.requireData("RP_META", "RP_POI", "RP_MODEL")) return; // the pipeline has not generated the data files yet

  const { clamp, round5k } = RP.analysis;
  const money = RP.money;
  const M = RP_MODEL;
  const META = RP_META;
  const thisYear = new Date().getFullYear();

  // ---------------------------------------------------------------- helpers
  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => [...document.querySelectorAll(selector)];

  // Numbers inside Hebrew (right-to-left) text must stay left-to-right, or "₪1,200" can show up mirrored
  const ltr = (text) => `<bdi dir="ltr">${text}</bdi>`;
  const signed = (n) => `${n > 0 ? "+" : n < 0 ? "\u2212" : "±"}${money(Math.abs(n))}`;
  const percent = (value, decimals = 0) => (value * 100).toFixed(decimals) + "%";

  /** Colours the filled part of a range slider (the CSS reads the --fill variable). */
  function paintSlider(input) {
    const paint = () => input.style.setProperty("--fill", ((input.value - input.min) / (input.max - input.min)) * 100 + "%");
    input.addEventListener("input", paint);
    paint();
  }

  /** Counts a number up to `to` (ease-out), starting from the value it showed before. */
  function animateNumber(element, to, duration = 650) {
    const from = Number(element.dataset.value || 0);
    const start = performance.now();
    function step(now) {
      const progress = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - progress, 3);
      element.textContent = RP.fmt(from + (to - from) * eased);
      if (progress < 1) requestAnimationFrame(step);
      else element.dataset.value = to;
    }
    requestAnimationFrame(step);
  }

  /** Plays a CSS animation again: remove the class, force a layout so the browser notices, add it back. */
  function restartAnimation(element, className) {
    element.classList.remove(className);
    void element.offsetWidth;
    element.classList.add(className);
  }

  // ---------------------------------------------------------------- dark / light theme
  // (index.html sets the saved theme before the page is drawn, to avoid a flash)
  $("#theme-toggle").onclick = () => {
    const root = document.documentElement;
    const next = root.getAttribute("data-theme") === "light" ? "dark" : "light";
    root.setAttribute("data-theme", next);
    try {
      localStorage.setItem("realprice-theme", next);
    } catch (e) {} // storage can be blocked (private mode); the theme just is not remembered
  };

  // ---------------------------------------------------------------- step 1: location (map + picker)
  let loc = null; // the chosen location {lat, lon, label, exact}, null until the user picks one

  const map = RP.createMap($("#map"), setLocation);
  const mapLayer = L.layerGroup().addTo(map); // the markers of the current property
  const picker = RP.initPicker($("#picker"), setLocation);

  function setLocation(chosen) {
    loc = chosen;
    RP.showLocationOnMap(map, mapLayer, chosen);
    picker.say("מיקום נבחר: " + chosen.label);
    $("#load-status").textContent = "";
  }

  // ---------------------------------------------------------------- step 2 + 3: property details
  const field = {
    area: $("#f-area"),
    rooms: $("#f-rooms"),
    year: $("#f-year"),
    yearUnknown: $("#f-year-unknown"),
    nature: $("#f-nature"),
    date: $("#f-date"),
    asking: $("#f-asking"),
  };

  field.nature.innerHTML = [...M.nature_categories, M.other_label]
    .map((nature) => `<option value="${nature}">${RP.natureLabel(nature)}</option>`)
    .join("");
  field.date.value = new Date().toISOString().slice(0, 10); // today
  field.year.max = thisYear + 3;

  // The text next to each slider
  const sliderText = {
    "f-area": (v) => `${v} מ״ר`,
    "f-rooms": (v) => v,
    "f-year": (v) => (field.yearUnknown.checked ? "לא ידועה" : v),
  };
  [field.area, field.rooms, field.year].forEach((slider) => {
    const output = $(`[data-out-for="${slider.id}"]`);
    const showValue = () => (output.textContent = sliderText[slider.id](slider.value));
    paintSlider(slider);
    slider.addEventListener("input", showValue);
    field.yearUnknown.addEventListener("change", showValue);
    showValue();
  });
  field.yearUnknown.addEventListener("change", () => {
    field.year.disabled = field.yearUnknown.checked;
  });

  /** The form as the `inputs` object that predictor.js and analysis.js expect. */
  function readInputs() {
    return {
      lat: loc.lat,
      lon: loc.lon,
      area: +field.area.value,
      rooms: +field.rooms.value,
      yearBuilt: field.yearUnknown.checked ? null : +field.year.value,
      nature: field.nature.value,
      date: new Date(field.date.value || Date.now()),
    };
  }

  // "Buy" mode asks "is it worth it?", "sell" mode asks "how much should I ask?"
  let mode = "buy";
  $$(".mode-toggle button").forEach((button) => {
    button.onclick = () => {
      $$(".mode-toggle button").forEach((b) => b.classList.remove("active"));
      button.classList.add("active");
      mode = button.dataset.mode;
      $("#submit-label").textContent = mode === "buy" ? "חשבו שווי שוק" : "חשבו מחיר מבוקש";
    };
  });

  // ---------------------------------------------------------------- the 3-step form
  const steps = $$(".form-step");
  const stepDots = $$(".step-dot");
  const form = $("#valuation-form");
  const formStatus = $("#load-status");
  let currentStep = 1;

  function showStep(direction) {
    steps.forEach((step) => {
      const isCurrent = +step.dataset.step === currentStep;
      step.classList.toggle("is-active", isCurrent);
      step.classList.toggle("dir-back", isCurrent && direction === "back"); // slide in from the other side
    });
    stepDots.forEach((dot) => {
      const n = +dot.dataset.dot;
      dot.classList.toggle("active", n === currentStep);
      dot.classList.toggle("done", n < currentStep);
      dot.setAttribute("aria-selected", n === currentStep);
    });
    $("#step-bar-fill").style.width = (currentStep / steps.length) * 100 + "%";
    // A map that was hidden while another step was shown must be told to measure itself again
    if (currentStep === 1) setTimeout(() => map.invalidateSize(), 60);
  }

  function goToStep(n, direction) {
    if (n > currentStep && currentStep === 1 && !loc) {
      // cannot leave step 1 without a location: shake the form and say why
      formStatus.textContent = "בחרו קודם מיקום (גוש/חלקה, כתובת או לחיצה על המפה).";
      restartAnimation(form, "shake");
      return;
    }
    formStatus.textContent = "";
    currentStep = n;
    showStep(direction);
  }

  $$(".step-next").forEach((button) => (button.onclick = () => goToStep(+button.dataset.goto, "next")));
  $$(".step-back").forEach((button) => (button.onclick = () => goToStep(+button.dataset.goto, "back")));
  stepDots.forEach((dot) => {
    dot.onclick = () => goToStep(+dot.dataset.dot, +dot.dataset.dot < currentStep ? "back" : "next");
    dot.onkeydown = (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        dot.onclick();
      }
    };
  });
  showStep();

  // ---------------------------------------------------------------- showing the results
  const chart = new RP.CurveChart($("#curve-svg"));
  $$(".axis-toggle[aria-label='ציר'] button").forEach((button) => {
    button.onclick = () => {
      $$(".axis-toggle[aria-label='ציר'] button").forEach((b) => b.classList.remove("active"));
      button.classList.add("active");
      chart.setAxis(button.dataset.axis);
    };
  });

  function showPrice(result) {
    animateNumber($("#price-value"), round5k(result.price));
    restartAnimation($("#price-value"), "bump");
    $("#range-low").textContent = money(round5k(result.low));
    $("#range-high").textContent = money(round5k(result.high));
    const rangePercent = Math.round((M.interval[1] - M.interval[0]) * 100);
    $("#range-note").textContent = `טווח סביר (${rangePercent}%) – אומדן סטטיסטי, לא שמאות.`;
  }

  function showReliability(result, inputs) {
    const rel = RP.analysis.reliability(result, loc, inputs);
    $("#coverage-level").textContent = rel.level;
    $("#coverage-fill").style.width = rel.score + "%";
    const approximate = loc.exact === false ? "; המיקום משוער (מרכז הגוש)" : "";
    $("#coverage-text").textContent = `רוחב הטווח הוא ${Math.round(rel.width * 100)}% מהמחיר${approximate}. מדד יוריסטי, לא רווח סמך סטטיסטי.`;
  }

  /** The deal-score card; shown only if the user entered an asking price. */
  function showDeal(estimate, asking) {
    const card = $("#deal-card");
    if (!(asking > 0)) {
      card.hidden = true;
      return;
    }
    const deal = RP.analysis.dealScore(estimate, asking);
    const gap = Math.round((Math.abs(estimate - asking) / estimate) * 100);
    card.hidden = false;
    card.className = "deal-card " + deal.cls;
    $("#deal-score").textContent = deal.score + " / 100";
    $("#deal-label").textContent = deal.label;
    $("#deal-fill").style.width = deal.score + "%";
    const who = mode === "sell" ? "המחיר שביקשתם" : "המוכר מבקש";
    const direction = asking < estimate ? "מתחת" : "מעל";
    $("#deal-explain").innerHTML = `${who} ${ltr(money(asking))} – ${gap}% ${direction} להערכת המודל.`;
  }

  /** Distance to the nearest school and stop, and to the centre of Tel Aviv. */
  function showSurroundings(result) {
    const tiles = result.near.layers.map(({ layer, nearest, count }) => {
      const distance = nearest ? RP.meters(nearest.d) : "-";
      const name = nearest ? (nearest.p.name || "ללא שם") + " · " : "";
      return `<div><strong>${distance}</strong><span>${layer.icon} ${layer.label} הקרוב</span><em>${name}${count} ברדיוס ${RP.meters(layer.radius)}</em></div>`;
    });
    tiles.push(`<div><strong>${RP.meters(result.near.distCenter)}</strong><span>🏙️ מרחק ממרכז תל אביב</span><em>קו אווירי</em></div>`);
    $("#near-grid").innerHTML = tiles.join("");
  }

  /** Where does the property stand among the same property at other places within 3 km? Plus the price curves. */
  function showMarket(inputs, result) {
    const points = RP.analysis.sweep(inputs, [600, 1500, 3000], 16);
    if (points.length < 6) {
      $("#percentile").textContent = "—";
      $("#position-title").textContent = "אין מספיק נקודות להשוואה";
      $("#position-text").textContent = "באזור הזה כמעט אין נקודות ידועות (בית ספר/תחנה) לשם השוואה.";
      $("#market-summary").textContent = "ההשוואה לא זמינה במיקום הזה.";
    } else {
      // percentile = the share of the other places where the same property is cheaper per sqm
      const cheaper = points.filter((p) => p.perSqm < result.perSqm).length;
      const pct = Math.round((cheaper / points.length) * 100);
      $("#percentile").textContent = pct;
      $("#position-marker").style.setProperty("--pos", pct + "%");
      $("#position-scope").textContent = "עד 3 ק״מ";
      $("#position-title").textContent = pct >= 70 ? "יקר מרוב הנקודות בסביבה" : pct <= 30 ? "זול מרוב הנקודות בסביבה" : "באמצע טווח המחירים באזור";
      $("#position-text").textContent = `מחיר המ״ר כאן גבוה מ-${pct}% מתוך ${points.length} נקודות בטווח 3 ק״מ, לאותו נכס בדיוק (לפי המודל).`;
      $("#market-summary").textContent = `${points.length} נקודות בסביבה הושוו לנכס שלכם, כולן מחושבות במודל.`;
    }
    chart.setData(RP.analysis.priceCurves(inputs, result));
  }

  const COMPASS = ["צפון", "צפון-מזרח", "מזרח", "דרום-מזרח", "דרום", "דרום-מערב", "מערב", "צפון-מערב"];

  /** Cards with the same property 1.5 km away in each of the 8 compass directions. */
  function showAround(inputs, result) {
    const grid = $("#similar-grid");
    const points = RP.analysis.sweep(inputs, [1500], 8);
    if (!points.length) {
      grid.innerHTML = '<div class="empty-card">לא נמצאו נקודות מאוישות בסביבה להשוואה.</div>';
      return;
    }
    grid.innerHTML = points
      .map((p) => {
        const diff = p.price - result.price;
        const places = p.near.layers.map((l) => l.nearest && `${l.layer.icon} ${RP.meters(l.nearest.d)}`).filter(Boolean).join(" · ");
        return `<article class="similar-card"><div class="similar-title"><strong>${COMPASS[p.k]}</strong><span>${RP.meters(p.d)}</span></div><div class="sub">${places || "—"}</div>
        <div class="similar-bottom"><strong>${ltr(money(round5k(p.price)))}</strong><span class="${diff >= 0 ? "above" : "below"}">${ltr(signed(round5k(diff)))} מההערכה</span></div></article>`;
      })
      .join("");
  }

  function showReceipt(inputs, result) {
    const { startPrice, rows } = RP.analysis.receipt(inputs, loc.label);
    $("#receipt-empty").hidden = true;
    $("#receipt").hidden = false;
    $("#receipt-car").textContent = `${inputs.area} מ״ר · ${inputs.rooms} חד׳`;
    $("#receipt-date").textContent = new Date().toLocaleDateString("he-IL", { year: "numeric", month: "short", day: "numeric" });
    $("#receipt-base-value").textContent = money(startPrice);
    $("#receipt-base-note").textContent = "(דירה טיפוסית: 90 מ״ר, 4 חדרים, מרכז תל אביב)";
    // each line appears 90 ms after the previous one
    $("#receipt-items").innerHTML = rows
      .map((row, i) => `<div class="receipt-item" style="animation-delay:${i * 90}ms"><div class="ri-text"><span class="ri-label">${row.label}</span><span class="ri-detail">${row.from} ← ${row.to}</span></div><span class="ri-amount ${row.delta >= 0 ? "pos" : "neg"}">${signed(row.delta)}</span></div>`)
      .join("");
    $("#receipt-total").textContent = money(result.price);
    $("#receipt-note").textContent = "הערכת הבסיס + כל השורות מעלה משחזרים בדיוק את חיזוי המודל (לפני עיגול).";
  }

  /** Bars with the price of the same property for each deal type. */
  function showNatureBars(inputs) {
    const rows = RP.analysis.priceByNature(inputs);
    const maxPrice = Math.max(...rows.map((r) => r.price)) || 1;
    $("#nature-bars").innerHTML =
      rows
        .map((r) => `<div class="bar-row wide"><span>${r.label}</span><div><i style="width:${(r.price / maxPrice) * 100}%"></i></div><b>${money(round5k(r.price))}</b></div>`)
        .join("") + '<p class="note">אותו שטח, חדרים ומיקום – רק סוג הנכס משתנה.</p>';
  }

  // ---------------------------------------------------------------- what-if sliders
  const whatIf = { area: $("#whatif-area"), rooms: $("#whatif-rooms"), year: $("#whatif-year") };
  const whatIfText = { area: $("#whatif-area-value"), rooms: $("#whatif-rooms-value"), year: $("#whatif-year-value") };
  // The year slider always has a position, but it only counts once the user has moved it (or entered a year)
  let whatIfYearUsed = false;
  const whatIfLabel = {
    area: (v) => `${v} מ״ר`,
    rooms: (v) => v,
    year: (v) => (whatIfYearUsed ? v : "לא ידועה"),
  };
  let lastResult = null; // {inputs, result} of the last estimate; the what-if sliders compare against it

  whatIf.year.max = thisYear + 3;
  Object.keys(whatIf).forEach((key) => {
    paintSlider(whatIf[key]);
    whatIfText[key].textContent = whatIfLabel[key](whatIf[key].value);

    whatIf[key].addEventListener("input", (e) => {
      // e.isTrusted is false for the events that showResult() fires by itself to refresh the labels
      if (key === "year" && e.isTrusted) whatIfYearUsed = true;
      whatIfText[key].textContent = whatIfLabel[key](whatIf[key].value);
      if (!lastResult || !e.isTrusted) return;

      const inputs = {
        ...lastResult.inputs,
        area: +whatIf.area.value,
        rooms: +whatIf.rooms.value,
        yearBuilt: whatIfYearUsed ? +whatIf.year.value : lastResult.inputs.yearBuilt,
      };
      const price = RP.estimate(inputs).price;
      const diff = price - lastResult.result.price;
      $("#whatif-new").textContent = money(round5k(price));
      restartAnimation($("#whatif-new"), "pulse");
      const note = $("#whatif-delta");
      note.innerHTML = `${ltr(signed(round5k(diff)))} לעומת ההערכה הנוכחית · הרצה אמיתית של המודל`;
      restartAnimation(note, "flash");
    });
  });

  /** Puts the values of the new estimate into the what-if sliders. */
  function resetWhatIf(inputs, result) {
    whatIf.area.value = inputs.area;
    whatIf.rooms.value = inputs.rooms;
    whatIfYearUsed = inputs.yearBuilt != null;
    whatIf.year.value = inputs.yearBuilt ?? 1995;
    Object.values(whatIf).forEach((slider) => slider.dispatchEvent(new Event("input"))); // refresh labels and fill
    $("#whatif-current").textContent = money(round5k(result.price));
    $("#whatif-new").textContent = money(round5k(result.price));
    $("#whatif-delta").textContent = "הזיזו סליידר כדי להריץ חיזוי נוסף של המודל.";
  }

  // ---------------------------------------------------------------- submit: calculate everything
  form.onsubmit = async (e) => {
    e.preventDefault();
    if (!loc) {
      goToStep(1, "back");
      formStatus.textContent = "בחרו קודם מיקום.";
      return;
    }
    formStatus.textContent = "מנתח נתוני שוק…";
    form.classList.add("is-loading");
    $(".calc-submit").classList.add("is-busy");
    // a short pause so the browser can draw the "loading" state before the (synchronous) calculation blocks it
    await new Promise((resolve) => setTimeout(resolve, 120));

    try {
      const inputs = readInputs();
      RP.showLocationOnMap(map, mapLayer, loc);
      const result = RP.predict(inputs);
      lastResult = { inputs, result };

      showPrice(result);
      showReliability(result, inputs);
      showDeal(result.price, +field.asking.value);
      showSurroundings(result);
      showMarket(inputs, result);
      showAround(inputs, result);
      showReceipt(inputs, result);
      showNatureBars(inputs);
      resetWhatIf(inputs, result);

      formStatus.textContent = "ההערכה הושלמה";
      // on a phone the result panel is below the form, so scroll to it
      if (window.innerWidth < 900) $(".result-panel").scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (error) {
      console.error(error);
      formStatus.textContent = "לא הצלחנו לחשב הערכה. בדקו את הפרטים ונסו שוב.";
    } finally {
      form.classList.remove("is-loading");
      $(".calc-submit").classList.remove("is-busy");
    }
  };

  // ---------------------------------------------------------------- numbers about the data and the model (these never change)
  const metrics = META.metrics; // the model's score on the test deals
  $("#dash-deals").textContent = RP.fmt(META.nDeals);
  $("#dash-parcels").textContent = RP.fmt(META.nParcels);
  $("#dash-schools").textContent = RP.fmt(RP_POI.schools.length);
  $("#dash-transit").textContent = RP.fmt(RP_POI.transit.length);
  $("#dash-r2").textContent = metrics.r2.toFixed(3);
  $("#dash-ape").textContent = percent(metrics.median_ape, 1);
  $("#accuracy-bars").innerHTML = [
    ["בטווח 10% מהמחיר", metrics.within_10pct],
    ["בטווח 20% מהמחיר", metrics.within_20pct],
    ["כיסוי הטווח המוצג", M.interval_coverage],
  ]
    .map(([label, share]) => `<div class="bar-row wide"><span>${label}</span><div><i style="width:${share * 100}%"></i></div><b>${percent(share)}</b></div>`)
    .join("");

  $("#stat-r2").textContent = metrics.r2.toFixed(4);
  $("#stat-mae").textContent = money(metrics.mae_nis);
  $("#stat-rmse").textContent = money(metrics.rmse_nis);
  $("#stat-rows").textContent = RP.fmt(M.n_train);
  $("#stat-trained").textContent = new Date(M.trained_at + "T00:00:00").toLocaleDateString("he-IL", { year: "numeric", month: "short", day: "numeric" });
  $("#stat-features").textContent = M.feature_names.length;
  $("#stat-trees").textContent = RP.fmt(M.models.median.roots.length);
  $("#stat-lo").textContent = Math.round(M.interval[0] * 100);
  $("#stat-hi").textContent = Math.round(M.interval[1] * 100);

  $("#importance-method").textContent = "תדירות השימוש בכל קבוצת גורמים בפיצולי העצים של מודל החציון – מדד גס, לא תרומה למחיר.";
  $("#importance-bars").innerHTML = RP.analysis
    .factorImportance()
    .map(({ name, percent: pct }) => `<div class="imp-row"><span>${name}</span><div class="imp-track"><i class="imp-fill" style="--pct:${pct}%"></i></div><b>${pct}%</b></div>`)
    .join("");

  // ---------------------------------------------------------------- page behaviour
  $("#live-dot").textContent = "המודל מוכן"; // replaces "loading model…" once everything above has run
  $("#cta-start").onclick = () => $("#valuation").scrollIntoView({ behavior: "smooth" });

  // Sections fade in the first time they scroll into view (the CSS animates the "in-view" class)
  const observer = new IntersectionObserver(
    (entries) => entries.forEach((entry) => entry.isIntersecting && entry.target.classList.add("in-view")),
    { threshold: 0.08 }
  );
  $$(".section,.hero-copy,.hero-visual").forEach((element) => observer.observe(element));
})();
