(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const state = {
    map: null, service: null, panorama: null, coverage: null,
    routeMode: false, pendingRoutePoint: false, routePoints: [], routePanoIds: [], routeLine: null, routeMarkers: [],
    panoramaTrace: null, scanRouteSamples: [], results: [], resultMarkers: [], selectedPanoId: null, mapsScriptAdded: false
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
        streetViewControl: true,
        streetViewControlOptions: { position: google.maps.ControlPosition.RIGHT_TOP },
        mapTypeControl: true, mapTypeControlOptions: { mapTypeIds: ["roadmap", "satellite", "hybrid", "terrain"] },
        fullscreenControl: true, clickableIcons: false, disableDoubleClickZoom: true
      });
      state.service = new google.maps.StreetViewService();
      state.panorama = new google.maps.StreetViewPanorama($("panorama"), {
        addressControl: true, linksControl: true, panControl: true,
        enableCloseButton: false, fullscreenControl: true
      });
      state.coverage = new google.maps.StreetViewCoverageLayer();
      state.coverage.setMap(state.map);
      state.map.addListener("click", (event) => {
        // A map click is useful only while drawing. Never turn arbitrary map clicks
        // into results: drawing clicks are verified against Street View coverage.
        if (state.routeMode) void addRoutePoint(event.latLng);
      });
      $("mapPlaceholder").classList.add("hidden");
      $("connectionLabel").textContent = "GOOGLE MAPS CONNECTED";
      $("connectionLabel").style.color = "#caff70";
      $("connectBtn").textContent = "Connected";
      $("drawBtn").disabled = false;
      $("connectBtn").disabled = false;
      toast("Use Pegman to reveal Google's blue Street View roads. Draw over those roads; accepted points snap to real panorama positions.");
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
    if (enabled) toast("Click directly on a blue Street View road. Each click is checked; uncovered clicks are rejected and valid points snap to the actual panorama location.");
  }

  async function addRoutePoint(clickedLatLng) {
    // Google exposes StreetViewCoverageLayer as a visual overlay, not as selectable
    // line geometry. Validate each map click against nearby panorama metadata and
    // snap the accepted vertex to Google's actual panorama coordinates.
    if (state.pendingRoutePoint) {
      toast("Checking the previous selection. Please wait.");
      return;
    }
    state.pendingRoutePoint = true;
    $("finishBtn").disabled = true;
    toast("Checking whether this point is on Street View coverage…");

    try {
      const PICK_RADIUS_METERS = 15;
      const response = await getPanoramaAt(clickedLatLng, PICK_RADIUS_METERS);
      if (!state.routeMode) return;

      if (response.status !== "OK" || !response.data || !response.data.location) {
        toast("Not selectable: no Street View panorama within 15 m. Click directly on a blue covered road.");
        return;
      }

      const data = response.data;
      const panoId = data.location.pano;
      const snappedPosition = data.location.latLng;
      const snapDistance = google.maps.geometry.spherical.computeDistanceBetween(clickedLatLng, snappedPosition);
      if (!panoId || snapDistance > PICK_RADIUS_METERS) {
        toast("Not selectable: the nearest panorama is too far away. Choose a blue coverage line.");
        return;
      }
      if (state.routePanoIds.includes(panoId)) {
        toast("That panorama is already part of your selected route. Choose another point along the blue line.");
        return;
      }

      // Store the real panorama position, never the arbitrary mouse-click location.
      state.routePanoIds.push(panoId);
      const panoramaItem = addPanorama(data);
      if (panoramaItem) {
        panoramaItem.selected = true;
        updatePanoramaMarker(panoramaItem);
      }
      state.routePoints.push(snappedPosition);

      if (state.routeLine) state.routeLine.setPath(state.routePoints);
      else {
        state.routeLine = new google.maps.Polyline({
          map: state.map, path: state.routePoints, clickable: false,
          strokeColor: "#caff70", strokeOpacity: 0.98, strokeWeight: 5, zIndex: 20
        });
      }
      const marker = new google.maps.Marker({
        map: state.map, position: snappedPosition, clickable: false,
        icon: { path: google.maps.SymbolPath.CIRCLE, scale: 4.5, fillColor: "#17200f",
          fillOpacity: 1, strokeColor: "#caff70", strokeWeight: 2 },
        zIndex: 30
      });
      state.routeMarkers.push(marker);
      renderResults();
      updateRouteStats();
      $("finishBtn").disabled = state.routePoints.length < 2;
      $("undoBtn").disabled = false;
      $("clearBtn").disabled = false;
      toast("Covered road point accepted and snapped to panorama " + panoId.slice(0, 12) + "…");
    } catch (error) {
      toast("Street View coverage check failed. Check your Maps API key and try again.");
    } finally {
      state.pendingRoutePoint = false;
      $("finishBtn").disabled = !state.routeMode || state.routePoints.length < 2;
    }
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
    const removedPanoId = state.routePanoIds.pop();
    if (removedPanoId && !state.routePanoIds.includes(removedPanoId)) {
      const resultIndex = state.results.findIndex((item) => item.panoId === removedPanoId);
      if (resultIndex >= 0) {
        if (state.results[resultIndex].marker) state.results[resultIndex].marker.setMap(null);
        state.results.splice(resultIndex, 1);
        renderResults();
      }
    }
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
    state.routePanoIds = [];
    state.scanRouteSamples = [];
    if (state.panoramaTrace) state.panoramaTrace.setMap(null);
    state.panoramaTrace = null;
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

  function panoramaMarkerIcon(selected) {
    return {
      path: google.maps.SymbolPath.CIRCLE,
      scale: selected ? 7 : 6,
      fillColor: selected ? "#caff70" : "#64748b",
      fillOpacity: 1,
      strokeColor: selected ? "#14200e" : "#e2e8f0",
      strokeWeight: 2
    };
  }

  function updatePanoramaMarker(item) {
    if (!item || !item.marker) return;
    item.marker.setIcon(panoramaMarkerIcon(item.selected));
    item.marker.setTitle((item.selected ? "Selected" : "Not selected") +
      " panorama · click to " + (item.selected ? "deselect " : "select ") + item.panoId);
  }

  function togglePanoramaSelection(item) {
    if (!item) return;
    item.selected = !item.selected;
    updatePanoramaMarker(item);
    showDetails(item);
    renderResults();
    toast((item.selected ? "Selected " : "Deselected ") + "panorama " + item.panoId.slice(0, 12) + "…");
  }

  function addPanorama(data, routeOrder) {
    if (!data || !data.location || !data.location.pano || !data.location.latLng) return null;
    const panoId = data.location.pano;
    const existing = state.results.find((item) => item.panoId === panoId);
    if (existing) {
      if (Number.isFinite(routeOrder)) existing.routeOrder = Math.min(existing.routeOrder, routeOrder);
      return existing;
    }
    const location = data.location.latLng;
    const item = {
      panoId: panoId, lat: location.lat(), lng: location.lng(),
      date: data.imageDate || "Not provided by Google",
      description: data.location.description || "Street View panorama",
      links: Array.isArray(data.links) ? data.links.length : 0,
      routeOrder: Number.isFinite(routeOrder) ? routeOrder : Number.MAX_SAFE_INTEGER,
      selected: true, marker: null
    };
    item.marker = new google.maps.Marker({
      map: state.map, position: location, title: "Selected panorama · click to deselect",
      icon: panoramaMarkerIcon(true), clickable: true, zIndex: 50
    });
    item.marker.addListener("click", () => togglePanoramaSelection(item));
    state.results.push(item);
    return item;
  }

  function getPanoramaById(panoId) {
    return new Promise((resolve) => {
      state.service.getPanorama({ pano: panoId }, (data, status) => {
        resolve({ data: data || null, status: status });
      });
    });
  }

  function distanceToScanRoute(position, samples) {
    let bestDistance = Infinity;
    let bestIndex = 0;
    for (let i = 0; i < samples.length; i++) {
      const distance = google.maps.geometry.spherical.computeDistanceBetween(position, samples[i]);
      if (distance < bestDistance) {
        bestDistance = distance;
        bestIndex = i;
      }
    }
    return { distance: bestDistance, index: bestIndex };
  }

  function redrawPanoramaTrace() {
    if (state.panoramaTrace) {
      state.panoramaTrace.setMap(null);
      state.panoramaTrace = null;
    }
    const ordered = state.results
      .filter((item) => Number.isFinite(item.routeOrder))
      .sort((a, b) => a.routeOrder - b.routeOrder);
    if (ordered.length > 1) {
      state.panoramaTrace = new google.maps.Polyline({
        map: state.map,
        path: ordered.map((item) => ({ lat: item.lat, lng: item.lng })),
        strokeColor: "#20b8ff", strokeOpacity: 0.9, strokeWeight: 4,
        clickable: false, zIndex: 25
      });
    }
  }

  async function scanRoute() {
    if (state.routePoints.length < 2) return;
    $("scanBtn").disabled = true;
    $("scanBtn").textContent = "Scanning…";
    $("scanProgress").classList.remove("hidden");
    $("resultsList").innerHTML = "";
    $("resultsSubtitle").textContent = "DISCOVERING PANORAMAS";
    $("detailsCard").classList.add("hidden");
    state.results.forEach((item) => { if (item.marker) item.marker.setMap(null); });
    state.results = [];
    if (state.panoramaTrace) state.panoramaTrace.setMap(null);
    state.panoramaTrace = null;

    const spacing = Number($("spacing").value);
    const radius = Number($("radius").value);
    state.scanRouteSamples = sampleRoute(state.routePoints, 8);
    const seedPoints = sampleRoute(state.routePoints, spacing);
    const routeCorridorMeters = Math.max(18, Math.min(25, radius));
    const maxPanoramas = 700;
    const queued = [];
    const visited = new Set();
    let checked = 0;

    $("progressCount").textContent = "0 / " + seedPoints.length;
    $("progressText").textContent = "Finding panorama sequences along selected roads…";
    $("progressBar").style.width = "0%";

    // Seed from several positions on the user's selected road.
    for (let i = 0; i < seedPoints.length; i += 4) {
      const batch = seedPoints.slice(i, i + 4);
      const responses = await Promise.all(batch.map((point) => getPanoramaAt(point, radius)));
      responses.forEach((response) => {
        checked += 1;
        if (response.status !== "OK" || !response.data || !response.data.location || !response.data.location.latLng) return;
        const panoId = response.data.location.pano;
        const routePosition = distanceToScanRoute(response.data.location.latLng, state.scanRouteSamples);
        if (routePosition.distance > routeCorridorMeters) return;
        addPanorama(response.data, routePosition.index);
        visited.add(panoId);
        (response.data.links || []).forEach((link) => {
          if (link && link.pano && !visited.has(link.pano)) {
            visited.add(link.pano);
            queued.push(link.pano);
          }
        });
      });
      $("progressCount").textContent = checked + " / " + seedPoints.length;
      $("progressText").textContent = state.results.length + " panorama IDs found; following connected Street View panoramas…";
      $("progressBar").style.width = Math.round((checked / seedPoints.length) * 45) + "%";
      renderResults();
    }

    // Follow adjacent panorama IDs; keep only those close to the selected route.
    let expanded = 0;
    while (queued.length && state.results.length < maxPanoramas && expanded < maxPanoramas * 3) {
      const batchIds = queued.splice(0, 4);
      const responses = await Promise.all(batchIds.map((id) => getPanoramaById(id)));
      responses.forEach((response) => {
        expanded += 1;
        if (response.status !== "OK" || !response.data || !response.data.location || !response.data.location.latLng) return;
        const data = response.data;
        const panoId = data.location.pano;
        const routePosition = distanceToScanRoute(data.location.latLng, state.scanRouteSamples);
        if (routePosition.distance > routeCorridorMeters) return;
        const wasKnown = state.results.some((item) => item.panoId === panoId);
        addPanorama(data, routePosition.index);
        if (!wasKnown) {
          (data.links || []).forEach((link) => {
            if (link && link.pano && !visited.has(link.pano)) {
              visited.add(link.pano);
              queued.push(link.pano);
            }
          });
        }
      });
      $("progressText").textContent = state.results.length + " unique panoramas · exploring connected views…";
      $("progressCount").textContent = expanded + " link checks";
      $("progressBar").style.width = (45 + Math.min(54, Math.round((expanded / (maxPanoramas * 3)) * 54))) + "%";
      renderResults();
    }

    state.results.sort((a, b) => a.routeOrder - b.routeOrder);
    redrawPanoramaTrace();
    $("resultsSubtitle").textContent = "SCAN COMPLETE · " + state.results.length + " UNIQUE PANORAMAS";
    $("scanBtn").disabled = false;
    $("scanBtn").innerHTML = "Scan selected route <span>→</span>";
    $("exportBtn").disabled = !state.results.some((item) => item.selected);
    renderResults();
    if (!state.results.length) {
      toast("No Street View panoramas found near this path. Draw closer to the blue roads or increase search radius.");
    } else {
      toast("Found " + state.results.length + " unique panoramas. Click each map marker or checkbox to select/deselect it.");
    }
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
        const item = state.results[Number(input.dataset.index)];
        item.selected = input.checked;
        updatePanoramaMarker(item);
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
  $("selectAllBtn").addEventListener("click", () => {
    state.results.forEach((item) => { item.selected = true; updatePanoramaMarker(item); });
    renderResults();
    toast("All discovered panorama markers selected.");
  });
  $("selectNoneBtn").addEventListener("click", () => {
    state.results.forEach((item) => { item.selected = false; updatePanoramaMarker(item); });
    renderResults();
    toast("All discovered panorama markers deselected.");
  });
  $("exportBtn").addEventListener("click", exportSelected);
  $("closePreview").addEventListener("click", () => {
    $("previewModal").classList.add("hidden");
    if (state.panorama) state.panorama.setVisible(false);
  });
  document.addEventListener("keydown", (event) => { if (event.key === "Escape") $("closePreview").click(); });
})();