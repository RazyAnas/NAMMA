# NAMMA Street View Selector

A standalone Google Maps JavaScript application for viewing the Google Street View coverage overlay, drawing campus road routes, checking panorama availability along the selected route, previewing individual panoramas, and exporting the drawn route with selected panorama IDs.

## Hosted application

https://anasrazy-namma-streetview-selector.hf.space/

## Features

- Google Maps 2D map centred on VIT Vellore.
- Google Street View coverage overlay (blue covered road segments).
- Click-to-draw route selection along covered roads.
- Route length and vertex count.
- Route sampling at a configurable distance interval and search radius.
- Street View metadata lookup at sample points; deduplication by panorama ID.
- Panorama markers, metadata inspection and individual Street View preview.
- Export of the user's route coordinates and selected panorama IDs as JSON.

## Run locally

Open \`site/index.html\` using a local static server, or use any static hosting provider. No build step is required.

## Google Cloud configuration

Create a Google Maps Platform API key, enable Maps JavaScript API and billing, and add HTTP referrer restrictions for the deployed application. Paste the key into the app's connection field. The key is not stored by the application.

## Export scope

The JSON export contains the user-drawn route and selected Google panorama IDs only. The application does not bulk-download, cache, save or export Google Street View imagery. Individual panoramas are viewed through Google's embedded Street View viewer. Review the applicable Google Maps Platform terms and policies before using the data in another workflow.
