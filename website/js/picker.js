/* Location picker (step 1): tab 1 = gush / chelka / sub-chelka, tab 2 = free-text address. Styled like the Auto-Price form. */
(function () {
  RP.createPicker = (root, onLocate) => {
    root.innerHTML = `
      <div class="axis-toggle wide" role="tablist">
        <button type="button" role="tab" class="active" data-tab="parcel">גוש / חלקה</button>
        <button type="button" role="tab" data-tab="address">כתובת</button>
      </div>
      <div data-form="parcel">
        <div class="field-row three">
          <div class="field"><label for="p-gush">גוש</label><input id="p-gush" type="number" min="1" inputmode="numeric"></div>
          <div class="field"><label for="p-chelka">חלקה</label><input id="p-chelka" type="number" min="1" inputmode="numeric"></div>
          <div class="field"><label for="p-sub">תת-חלקה</label><input id="p-sub" type="number" min="0" inputmode="numeric" placeholder="לא חובה"></div>
        </div>
      </div>
      <div data-form="address" hidden>
        <div class="field"><label for="p-address">כתובת (רחוב, מספר, עיר)</label><input id="p-address" type="text" autocomplete="street-address" placeholder="לדוגמה: דיזנגוף 50 תל אביב"></div>
      </div>
      <button type="button" class="btn btn-primary picker-btn" id="p-go">אתר על המפה</button>
      <p class="picker-status" role="status" aria-live="polite"></p>`;
    const status = root.querySelector(".picker-status");
    const say = (msg, error = false) => { status.textContent = msg; status.classList.toggle("error", error); };
    const val = (id) => root.querySelector("#" + id).value.trim();
    let tab = "parcel";

    root.querySelectorAll("[data-tab]").forEach((btn) => btn.addEventListener("click", () => {
      tab = btn.dataset.tab;
      root.querySelectorAll("[data-tab]").forEach((b) => b.classList.toggle("active", b === btn));
      root.querySelectorAll("[data-form]").forEach((f) => (f.hidden = f.dataset.form !== tab));
      say("");
    }));

    const handlers = {
      parcel: async () => {
        const gush = +val("p-gush"), chelka = +val("p-chelka"), sub = val("p-sub");
        if (!gush || !chelka) return say("הזינו גוש וחלקה.", true);
        const hit = await RP.findParcel(gush, chelka);
        if (!hit) return say(`גוש ${gush} לא נמצא במאגר.`, true);
        say(hit.exact ? "" : `חלקה ${chelka} לא נמצאה - מוצג מרכז גוש ${gush}.`);
        onLocate({ lat: hit.lat, lon: hit.lon, label: `גוש ${gush} חלקה ${chelka}` + (sub ? ` / ${sub}` : ""), exact: hit.exact });
      },
      address: async () => {
        if (!val("p-address")) return say("הזינו כתובת.", true);
        const hit = await RP.geocode(val("p-address"));
        if (!hit) return say("הכתובת לא נמצאה, נסו לדייק (רחוב, מספר, עיר).", true);
        onLocate({ lat: hit.lat, lon: hit.lon, label: hit.label, exact: true });
      },
    };
    const run = async () => {
      say("מחפש...");
      try { await handlers[tab](); } catch (err) { say("החיפוש נכשל - בדקו את החיבור לאינטרנט ונסו שוב.", true); }
    };
    root.querySelector("#p-go").addEventListener("click", run);
    root.querySelectorAll("input").forEach((i) => i.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); run(); } }));
    return { say };
  };
})();
