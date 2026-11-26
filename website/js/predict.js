/* predict.html: choose a location, describe the property, get an estimated price and range. */
(function () {
  if (!RP.requireData("RP_META", "RP_POI", "RP_MODEL")) return;
  const M = RP_MODEL, today = new Date();
  let loc = null;
  const map = RP.createMap(document.getElementById("map"), setLocation);
  const marker = L.layerGroup().addTo(map);
  const picker = RP.createPicker(document.getElementById("picker"), setLocation);
  const form = document.getElementById("property"), out = document.getElementById("estimate");

  form.nature.innerHTML = [...M.nature_categories, M.other_label].map((c) => `<option>${c}</option>`).join("");
  form.date.value = today.toISOString().slice(0, 10);
  form.year.max = today.getFullYear() + 3;

  function setLocation(l) {
    loc = l;
    RP.showLocationOnMap(map, marker, l);
    picker.say("מיקום נבחר: " + l.label);
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    if (!loc) return picker.say("בחרו קודם מיקום (גוש/חלקה, כתובת או לחיצה על המפה).", true);
    RP.showLocationOnMap(map, marker, loc);

    const r = RP.predict({
      lat: loc.lat, lon: loc.lon, area: +form.area.value, rooms: +form.rooms.value,
      yearBuilt: form.year.value === "" ? null : +form.year.value, nature: form.nature.value, date: new Date(form.date.value)
    });
    const span = r.high - r.low || 1, pos = Math.min(100, Math.max(0, ((r.price - r.low) / span) * 100));
    const m = M.metrics;
    out.innerHTML = `
      <h2>הערכת מחיר</h2>
      <p class="hint">${loc.label}</p>
      <div class="figure">${RP.money(Math.round(r.price / 5000) * 5000)}</div>
      <div class="range"><i style="inset-inline-start:${pos}%"></i></div>
      <div class="range-labels"><span>${RP.money(Math.round(r.low / 5000) * 5000)}</span><span>טווח סביר (${Math.round((M.interval[1] - M.interval[0]) * 100)}%)</span><span>${RP.money(Math.round(r.high / 5000) * 5000)}</span></div>
      <div class="facts">
        <div><b>${RP.money(Math.round(r.perSqm / 100) * 100)}</b>למ״ר</div>
        ${r.near.layers.map(({ layer, nearest }) => `<div><b>${nearest ? RP.meters(nearest.d) : "-"}</b>${layer.label} הקרוב</div>`).join("")}
      </div>
      <p class="fineprint">המודל אומן על ${RP.fmt(M.n_train)} עסקאות (${M.trained_at}). בבדיקה על עסקאות שלא ראה, חציון הסטייה היה ${(m.median_ape * 100).toFixed(1)}% ו-${Math.round(m.within_20pct * 100)}% מההערכות היו בטווח של 20% מהמחיר האמיתי. זו הערכה סטטיסטית, לא שמאות.</p>`;
    out.hidden = false;
    out.scrollIntoView({ behavior: "smooth", block: "nearest" });
  });
})();