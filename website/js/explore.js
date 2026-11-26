/* index.html: locate a parcel / address and show the nearest school + public transport on the map. */
(function () {
  if (!RP.requireData("RP_META", "RP_POI")) return;
  const map = RP.createMap(document.getElementById("map"), (loc) => show(loc));
  const layer = L.layerGroup().addTo(map);
  const list = document.getElementById("results");

  function row({ icon, title, sub, value }) {
    return `<li><span class="ico">${icon}</span><div><b>${title}</b><small>${sub}</small></div><span class="dist">${value}</span></li>`;
  }

  function show(loc) {
    const near = RP.showLocationOnMap(map, layer, loc);
    const rows = near.layers.map(({ layer: meta, nearest, count }) => {
      if (!nearest) return row({ icon: meta.icon, title: meta.label, sub: "לא נמצא במאגר", value: "-" });
      const { p, d } = nearest;
      return row({
        icon: meta.icon,
        title: `${meta.label} הקרוב`,
        sub: `${p.name || "ללא שם"} · ${count} ברדיוס ${RP.meters(meta.radius)}`,
        value: RP.meters(d)
      });
    });
    rows.push(row({ icon: "🏙️", title: "מרחק ממרכז תל אביב", sub: "קו אווירי", value: RP.meters(near.distCenter) }));
    list.innerHTML = rows.join("");
  }

  RP.createPicker(document.getElementById("picker"), show);
})();