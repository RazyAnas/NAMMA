(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const state = {
    map: null, service: null, panorama: null, coverage: null,
    routeMode: false, routePoints: [], routeLine: null, routeMarkers: [],
    results: [], resultMarkers: [], selectedPanoId: null, mapsScriptAdded: false
  };
  const VIT = { lat: 12.9698, lng: 79.1559 };

  function setError(message) {
    $("mapError").textContent = message;
    $("mapError").classList.remove("hidden");
  }
  function clearError() {
    $("mapError").textContent = "";
    $("mapError").classList.add("hidden");
  }
  function toast(message) {
    $("mapToast").textContent = message;
    $("mapToast").classList.remove("hidden");
    window.clearTimeout(toast.timer);
    toast.timer = window.setTimeout(() => $("mapToast").classList.add("hidden"), 4200);
  }
  function escapeHtml(value) {
    return String(value == null ? "" : value).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    })[c]);
  }
  function loadGoogleMaps() {
    clearError();
    const key = $("apiKey").value.trim();
    if (!key) return setError("Paste your Google Maps JavaScript API key first.");
    if (window.google && window.google.maps) return initMap();
    if (state.mapsScriptAdded) return setError("Google Maps is still loading. Please wait a moment.");
    state.mapsScriptAdded = true;
    window.initNammaGoogleMaps = initMap;
    const script = document.createElement("script");
    script.src = "https://maps.googleapis.com/maps/api/js?key=" + encodeURIComponent(key) +
      "&callback=initNammaGoogleMaps&libraries=geometry,streetView&v=weekly";
    script.async = true;
    script.defer = true;
    script.onerror = () => {
      state.mapsScriptAdded = false;
      $("connectBtn").disabled = false;
      $("connectBtn").textContent = "Connect";
      setError("Google Maps could not load. Check your key, billing, HTTP referrer restrictions and Maps JavaScript API access.");
    };
    document.head.appendChild(script);
    $("connectBtn").disabled = true;
    $("connectBtn").textContent = "Loading…";
  }
  window.initNammaGoogleMaps = initMap;

  function initMap() {
    if (state.map) return;
    try {
      state.map = new google.maps.Map($("map"), {
        center: VIT, zoom: 16, mapTypeId: "roadmap",
        streetViewControl: false, mapTypeControl: true, fullscreenControl: true,
        clickableIcons: false, disableDoubleClickZoom: true
      });
      state.service = new google.maps.StreetViewService();
      state.panorama = new google.maps.StreetViewPanorama($("panorama"), {
        addressControl: true, linksControl: true, panControl: true,
        enableCloseButton: false, fullscreenControl: true
      });
      state.coverage = new google.maps.StreetViewCoverageLayer();
      state.coverage.setMap(state.map);
      state.map.addListener("click", (event) => {
        if (state.routeMode) addRoutePoint(event.latLng);
        else inspectPoint(event.latLng);
      });
      $("mapPlaceholder").classList.add("hidden");
      $("connectionLabel").textContent = "GOOGLE MAPS CONNECTED";
      $("connectionLabel").style.color = "#caff70";
      $("connectBtn").textContent = "Connected";
      $("drawBtn").disabled = false;
      $("connectBtn").disabled = false;
      toast("Blue road lines show Street View coverage. Draw only along the covered roads.");
    } catch (error) {
      $("connectBtn").disabled = false;
      $("connectBtn").textContent = "Connect";
      setError("Google Maps loaded, but Street View could not initialize: " + (error && error.message ? error.message : String(error)));
    }
  }

  function setRouteMode(enabled) {
    state.routeMode = enabled;
    $("drawBtn").textContent = enabled ? "Drawing active" : "＋ Start drawing";
    $("drawBtn").classList.toggle("button-primary", enabled);
    $("finishBtn").disabled = !enabled || state.routePoints.length < 2;
    $("undoBtn").disabled = state.routePoints.length === 0;
    $("clearBtn").disabled = state.routePoints.length === 0;
    $("drawHint").textContent = enabled
      ? "Click along the blue Street View coverage lines to add route points. Press Finish when done."
      : "Connect the map, press Start drawing, then click along the blue roads. Press Finish when the route is complete.";
    if (enabled) toast("Route drawing enabled. Click the map to add points.");
  }

  function addRoutePoint(latLng) {
    state.routePoints.push(latLng);
    if (state.routeLine) state.routeLine.setPath(state.routePoints);
    else {
      state.routeLine = new google.maps.Polyline({
        map: state.map, path: state.routePoints, clickable: false,
        strokeColor: "#caff70", strokeOpacity: 0.98, strokeWeight: 5, zIndex: 20
      });
    }
    const marker = new google.maps.Marker({
      map: state.map, position: latLng, clickable: false,
      icon: { path: google.maps.SymbolPath.CIRCLE, scale: 4.5, fillColor: "#17200f",
        fillOpacity: 1, strokeColor: "#caff70", strokeWeight: 2 },
      zIndex: 30
    });
    state.routeMarkers.push(marker);
    updateRouteStats();
    $("finishBtn").disabled = state.routePoints.length < 2;
    $("undoBtn").disabled = false;
    $("clearBtn").disabled = false;
  }

  function updateRouteStats() {
    const meters = state.routePoints.length > 1
      ? google.maps.geometry.spherical.computeLength(state.routePoints) : 0;
    $("routeDistance").textContent = meters >= 1000 ? (meters / 1000).toFixed(2) + " km" : Math.round(meters) + " m";
    $("routeVertices").textContent = String(state.routePoints.length);
    $("scanBtn").disabled = !state.map || state.routePoints.length < 2 || state.routeMode;
  }

  function undoPoint() {
    if (!state.routePoints.length) return;
    state.routePoints.pop();
    const marker = state.routeMarkers.pop();
    if (marker) marker.setMap(null);
    if (state.routeLine) {
      if (state.routePoints.length < 2) {
        state.routeLine.setMap(null);
        state.routeLine = null;
      } else state.routeLine.setPath(state.routePoints);
    }
    updateRouteStats();
    $("finishBtn").disabled = state.routePoints.length < 2;
    $("undoBtn").disabled = state.routePoints.length === 0;
    $("clearBtn").disabled = state.routePoints.length === 0;
  }

  function clearRoute() {
    state.routeMarkers.forEach((marker) => marker.setMap(null));
    if (state.routeLine) state.routeLine.setMap(null);
    state.routeMarkers = [];
    state.routeLine = null;
    state.routePoints = [];
    state.results.forEach((item) => { if (item.marker) item.marker.setMap(null); });
    state.results = [];
    renderResults();
    setRouteMode(false);
    updateRouteStats();
    $("scanProgress").classList.add("hidden");
    $("resultsSubtitle").textContent = "WAITING FOR A SCAN";
    $("exportBtn").disabled = true;
  }

  function sampleRoute(path, spacing) {
    const lengths = [];
    let total = 0;
    for (let i = 0; i < path.length - 1; i++) {
      const len = google.maps.geometry.spherical.computeDistanceBetween(path[i], path[i + 1]);
      lengths.push(len);
      total += len;
    }
    if (!total) return path.slice(0, 1);
    let actualSpacing = spacing;
    if (Math.ceil(total / actualSpacing) > 300) actualSpacing = total / 300;
    const distances = [];
    for (let d = 0; d < total; d += actualSpacing) distances.push(d);
    distances.push(total);
    return distances.map((distance) => {
      let traversed = 0;
      for (let i = 0; i < lengths.length; i++) {
        if (distance <= traversed + lengths[i] || i === lengths.length - 1) {
          const fraction = lengths[i] ? Math.max(0, Math.min(1, (distance - traversed) / lengths[i])) : 0;
          return google.maps.geometry.spherical.interpolate(path[i], path[i + 1], fraction);
        }
        traversed += lengths[i];
      }
      return path[path.length - 1];
    });
  }

  function getPanoramaAt(position, radius) {
    return new Promise((resolve) => {
      state.service.getPanorama({
        location: position, radius: radius,
        preference: google.maps.StreetViewPreference.NEAREST
      }, (data, status) => resolve({ data: data || null, status: status }));
    });
  }

  function addPanorama(data) {
    if (!data || !data.location || !data.location.pano) return null;
    const panoId = data.location.pano;
    const existing = state.results.find((item) => item.panoId === panoId);
    if (existing) return existing;
    const location = data.location.latLng;
    const item = {
      panoId: panoId, lat: location.lat(), lng: location.lng(),
      date: data.imageDate || "Not provided by Google",
      description: data.location.description || "Street View panorama",
      links: Array.isArray(data.links) ? data.links.length : 0,
      selected: true, marker: null
    };
    item.marker = new google.maps.Marker({
      map: state.map, position: location, title: "Street View panorama: " + panoId,
      icon: { path: google.maps.SymbolPath.CIRCLE, scale: 6,
        fillColor: "#caff70", fillOpacity: 1, strokeColor: "#182114", strokeWeight: 2 },
      zIndex: 40
    });
    item.marker.addListener("click", () => showDetails(item));
    state.results.push(item);
    return item;
  }

  async function scanRoute() {
    if (state.routePoints.length < 2) return;
    $("scanBtn").disabled = true;
    $("scanBtn").textContent = "Scanning…";
    $("scanProgress").classList.remove("hidden");
    $("resultsList").innerHTML = "";
    $("resultsSubtitle").textContent = "SCANNING ROUTE";
    $("detailsCard").classList.add("hidden");
    state.results.forEach((item) => { if (item.marker) item.marker.setMap(null); });
    state.results = [];
    const spacing = Number($("spacing").value);
    const radius = Number($("radius").value);
    const samples = sampleRoute(state.routePoints, spacing);
    let checked = 0;
    $("progressCount").textContent = "0 / " + samples.length;
    $("progressText").textContent = "Checking panorama coverage…";
    $("progressBar").style.width = "0%";
    for (let i = 0; i < samples.length; i += 4) {
      const batch = samples.slice(i, i + 4);
      const responses = await Promise.all(batch.map((point) => getPanoramaAt(point, radius)));
      responses.forEach((response) => {
        checked += 1;
        if (response.status === "OK" && response.data && response.data.location) addPanorama(response.data);
      });
      $("progressCount").textContent = checked + " / " + samples.length;
      $("progressText").textContent = state.results.length + " unique panoramas found";
      $("progressBar").style.width = Math.round((checked / samples.length) * 100) + "%";
      renderResults();
    }
    $("resultsSubtitle").textContent = "SCAN COMPLETE";
    $("scanBtn").disabled = false;
    $("scanBtn").innerHTML = "Scan selected route <span>→</span>";
    $("exportBtn").disabled = state.results.length === 0;
    renderResults();
    if (!state.results.length) toast("No panoramas found on this route. Try a larger search radius or check that your path follows blue coverage.");
    else toast("Found " + state.results.length + " unique panorama IDs along " + samples.length + " checked points.");
  }

  function renderResults() {
    $("foundCount").textContent = String(state.results.length);
    $("selectedCount").textContent = String(state.results.filter((item) => item.selected).length);
    $("selectAllBtn").disabled = !state.results.length;
    $("selectNoneBtn").disabled = !state.results.length;
    $("exportBtn").disabled = !state.results.some((item) => item.selected);
    if (!state.results.length) {
      $("resultsList").innerHTML = '<div class="empty-state">No panorama results yet. Draw a covered road and scan it to discover available panoramas.</div>';
      return;
    }
    $("resultsList").innerHTML = state.results.map((item, index) =>
      '<div class="pano-row"><input type="checkbox" data-index="' + index + '" ' + (item.selected ? "checked" : "") +
      ' aria-label="Select panorama"><div class="pano-main" data-open="' + index + '"><b>' + escapeHtml(item.panoId) +
      '</b><span>' + escapeHtml(item.date) + ' · ' + item.lat.toFixed(5) + ', ' + item.lng.toFixed(5) + '</span></div>' +
      '<button type="button" data-preview="' + index + '">Preview</button></div>'
    ).join("");
    $("resultsList").querySelectorAll("input[data-index]").forEach((input) => {
      input.addEventListener("change", () => {
        state.results[Number(input.dataset.index)].selected = input.checked;
        renderResults();
      });
    });
    $("resultsList").querySelectorAll("[data-open]").forEach((node) => {
      node.addEventListener("click", () => showDetails(state.results[Number(node.dataset.open)]));
    });
    $("resultsList").querySelectorAll("[data-preview]").forEach((node) => {
      node.addEventListener("click", () => openPreview(state.results[Number(node.dataset.preview)]));
    });
  }

  function showDetails(item) {
    if (!item) return;
    state.selectedPanoId = item.panoId;
    $("detailsCard").classList.remove("hidden");
    $("detailsCard").innerHTML =
      '<div class="detail-title">Panorama details</div><div class="detail-grid">' +
      '<div><span>PANORAMA ID</span><b>' + escapeHtml(item.panoId) + '</b></div>' +
      '<div><span>CAPTURE DATE</span><b>' + escapeHtml(item.date) + '</b></div>' +
      '<div><span>LOCATION</span><b>' + escapeHtml(item.description) + '</b></div>' +
      '<div><span>COORDINATES</span><b>' + item.lat.toFixed(7) + ', ' + item.lng.toFixed(7) + '</b></div>' +
      '<div><span>CONNECTED PANORAMA LINKS</span><b>' + item.links + '</b></div></div>';
  }

  function openPreview(item) {
    if (!item || !state.panorama) return;
    showDetails(item);
    $("previewSubtitle").textContent = item.panoId;
    $("previewModal").classList.remove("hidden");
    state.panorama.setPano(item.panoId);
    state.panorama.setVisible(true);
  }

  function inspectPoint(position) {
    getPanoramaAt(position, Number($("radius").value)).then((response) => {
      if (response.status !== "OK" || !response.data) {
        toast("No Street View panorama found near this map point.");
        return;
      }
      const item = addPanorama(response.data);
      renderResults();
      showDetails(item);
      toast("Street View panorama found. It has been added to the results.");
    }).catch(() => toast("Could not inspect this point. Check the Maps API configuration."));
  }

  function exportSelected() {
    const ids = state.results.filter((item) => item.selected).map((item) => item.panoId);
    if (!ids.length) return;
    const payload = {
      format: "namma-streetview-route-v1",
      created_at: new Date().toISOString(),
      route: {
        type: "LineString",
        coordinates: state.routePoints.map((point) => [point.lng(), point.lat()])
      },
      panorama_ids: ids
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "vit-streetview-route-selection.json";
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  }

  $("connectBtn").addEventListener("click", loadGoogleMaps);
  $("apiKey").addEventListener("keydown", (event) => { if (event.key === "Enter") loadGoogleMaps(); });
  $("drawBtn").addEventListener("click", () => { if (state.map && !state.routeMode) setRouteMode(true); });
  $("finishBtn").addEventListener("click", () => {
    if (state.routePoints.length < 2) return;
    setRouteMode(false);
    $("scanBtn").disabled = false;
    toast("Route ready. Scan it to find unique Street View panoramas.");
  });
  $("undoBtn").addEventListener("click", undoPoint);
  $("clearBtn").addEventListener("click", clearRoute);
  $("scanBtn").addEventListener("click", scanRoute);
  $("selectAllBtn").addEventListener("click", () => { state.results.forEach((item) => item.selected = true); renderResults(); });
  $("selectNoneBtn").addEventListener("click", () => { state.results.forEach((item) => item.selected = false); renderResults(); });
  $("exportBtn").addEventListener("click", exportSelected);
  $("closePreview").addEventListener("click", () => {
    $("previewModal").classList.add("hidden");
    if (state.panorama) state.panorama.setVisible(false);
  });
  document.addEventListener("keydown", (event) => { if (event.key === "Escape") $("closePreview").click(); });
})();