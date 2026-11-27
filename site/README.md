---
title: NAMMA
emoji: 🏠
colorFrom: red
colorTo: yellow
sdk: static
pinned: true
short_description: Turn a phone video into a 3D tour you can walk through
hf_oauth: true
hf_oauth_expiration_minutes: 480
hf_oauth_scopes:
  - read-repos
  - write-repos
  - manage-repos
  - write-discussions
---

# NAMMA

Film a room with your phone and get a 3D tour you can walk through in the browser.

- The training runs on your own free Kaggle GPU. A room takes about an hour.
- The video and the tour are stored in your own Hugging Face account.
- This Space is a static page. It has no server and stores nothing.

Signing in asks for permission to read, write and manage your own repositories (to create the NAMMA datasets and tour Space in
your account) and to open a pull request for you (only when you submit a tour to the public list).

Source code: https://github.com/RazyAnas/NAMMA

This folder is deployed by `scripts/site_deploy.py`, which also writes `gallery.json` from `gallery.src.json`.
