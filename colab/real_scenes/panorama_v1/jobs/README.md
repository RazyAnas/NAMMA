# Panorama job manifests

The workflow triggers only when a JSON job request is added or modified in this directory. Do not put source code or general configuration here.

Use `submit_job.py` to upload a bundle and create a unique job manifest. Each job is a new file and has a unique `job_id`; the workflow creates a dedicated Kaggle dataset and kernel for that job.
