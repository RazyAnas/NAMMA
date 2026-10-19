// Client for the laptop's /api/v1/remote endpoints (app/pipeline/stages/phone.py): pairing check,
// resumable chunked upload, render polling and result download. Chunks stay under the 100 MB
// request limit of a Cloudflare quick tunnel, and a dropped chunk is retried, not the whole file.

export class RenderClient {
  constructor(server, key) {
    this.server = server.replace(/\/+$/, '');
    this.key = key || '';
  }

  async call(path, options = {}) {
    const headers = { ...(options.headers || {}) };
    if (this.key) headers['X-Namma-Key'] = this.key;
    const response = await fetch(this.server + path, { ...options, headers });
    if (!response.ok) {
      let detail = `${response.status} ${response.statusText}`;
      try {
        const body = await response.json();
        detail = typeof body.detail === 'string' ? body.detail : body.detail?.message || JSON.stringify(body.detail);
      } catch {
        // not JSON
      }
      const error = new Error(detail);
      error.status = response.status;
      throw error;
    }
    return response;
  }

  async json(path, options = {}) {
    return (await this.call(path, options)).json();
  }

  hello() {
    return this.json('/api/v1/remote/hello');
  }

  // Sends the blob in chunks; onProgress(fraction). Returns {render_id, project_id, kind}.
  async send(blob, filename, { quality = 'balanced', name, onProgress = () => {}, retries = 4 } = {}) {
    const start = await this.json('/api/v1/remote/uploads', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename, size: blob.size, name: name || filename }),
    });
    const size = start.chunk_bytes;
    for (let index = 0; index < start.chunks; index++) {
      const part = blob.slice(index * size, Math.min(blob.size, (index + 1) * size));
      for (let attempt = 0; ; attempt++) {
        try {
          await this.call(`/api/v1/remote/uploads/${start.upload_id}/chunks/${index}`, { method: 'PUT', body: part });
          break;
        } catch (error) {
          if (attempt >= retries || error.status === 401) throw error;
          await new Promise((resolve) => setTimeout(resolve, 1000 * 2 ** attempt));
        }
      }
      onProgress((index + 1) / start.chunks);
    }
    return this.json(`/api/v1/remote/uploads/${start.upload_id}/finish`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ quality }),
    });
  }

  status(renderId) {
    return this.json(`/api/v1/remote/renders/${renderId}`);
  }

  async file(renderId, name) {
    return (await this.call(`/api/v1/remote/renders/${renderId}/files/${name}`)).arrayBuffer();
  }

  // Polls until the render finishes; onUpdate(status) on every poll.
  retry(renderId) {
    return this.json(`/api/v1/remote/renders/${renderId}/retry`, { method: 'POST' });
  }

  async wait(renderId, onUpdate = () => {}, intervalMs = 3000) {
    for (;;) {
      let status;
      try {
        status = await this.status(renderId);
      } catch (error) {
        if (error.status === 401 || error.status === 404) throw error;
        onUpdate({ status: 'unreachable', error: error.message });
        await new Promise((resolve) => setTimeout(resolve, intervalMs * 2));
        continue;
      }
      onUpdate(status);
      if (status.status === 'succeeded' || status.status === 'failed') return status;
      await new Promise((resolve) => setTimeout(resolve, intervalMs));
    }
  }
}

// ?server=...&key=... from the link the laptop prints, else what this phone saved last time.
export function pairingFromLocation(location, storage) {
  const params = new URLSearchParams(location.search);
  let server = params.get('server');
  let key = params.get('key');
  if (server) {
    try {
      storage?.setItem('namma.server', server);
      storage?.setItem('namma.key', key || '');
    } catch {
      // storage blocked: pairing lasts for this page only
    }
  } else {
    try {
      server = storage?.getItem('namma.server') || null;
      key = key || storage?.getItem('namma.key') || null;
    } catch {
      server = null;
    }
  }
  if (!server && location.protocol !== 'file:' && location.pathname.startsWith('/record')) {
    server = location.origin; // the page itself is served by the laptop
  }
  return { server, key: key || '' };
}
