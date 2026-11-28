/* Location picker (step 1 of the form). Two tabs: "gush / chelka" and "address".
   The markup is in index.html (inside <div id="picker">); this file only adds the behaviour. */
(function () {
  /**
   * @param root      the #picker element
   * @param onLocate  called with {lat, lon, label, exact} when a location was found
   * @returns         {say(message, isError)} - shows a status line under the form
   */
  RP.initPicker = (root, onLocate) => {
    const statusLine = root.querySelector(".picker-status");
    const tabButtons = root.querySelectorAll("[data-tab]");
    const tabForms = root.querySelectorAll("[data-form]");
    let currentTab = "parcel";

    function say(message, isError = false) {
      statusLine.textContent = message;
      statusLine.classList.toggle("error", isError);
    }
    const value = (id) => root.querySelector("#" + id).value.trim();

    // ---- switching between the two tabs
    tabButtons.forEach((button) =>
      button.addEventListener("click", () => {
        currentTab = button.dataset.tab;
        tabButtons.forEach((b) => b.classList.toggle("active", b === button));
        tabForms.forEach((form) => (form.hidden = form.dataset.form !== currentTab));
        say("");
      })
    );

    // ---- one search function per tab
    async function searchParcel() {
      const gush = +value("p-gush");
      const chelka = +value("p-chelka");
      const subChelka = value("p-sub"); // optional, only shown in the label
      if (!gush || !chelka) return say("הזינו גוש וחלקה.", true);

      const hit = await RP.findParcel(gush, chelka);
      if (!hit) return say(`גוש ${gush} לא נמצא במאגר.`, true);

      say(hit.exact ? "" : `חלקה ${chelka} לא נמצאה - מוצג מרכז גוש ${gush}.`);
      const label = `גוש ${gush} חלקה ${chelka}` + (subChelka ? ` / ${subChelka}` : "");
      onLocate({ lat: hit.lat, lon: hit.lon, label, exact: hit.exact });
    }

    async function searchAddress() {
      const address = value("p-address");
      if (!address) return say("הזינו כתובת.", true);

      const hit = await RP.geocode(address);
      if (!hit) return say("הכתובת לא נמצאה, נסו לדייק (רחוב, מספר, עיר).", true);
      onLocate({ lat: hit.lat, lon: hit.lon, label: hit.label, exact: true });
    }

    async function search() {
      say("מחפש...");
      try {
        await (currentTab === "parcel" ? searchParcel() : searchAddress());
      } catch (error) {
        say("החיפוש נכשל - בדקו את החיבור לאינטרנט ונסו שוב.", true);
      }
    }

    root.querySelector("#p-go").addEventListener("click", search);
    root.querySelectorAll("input").forEach((input) =>
      input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
          e.preventDefault(); // do not submit the big form behind the picker
          search();
        }
      })
    );

    return { say };
  };
})();
