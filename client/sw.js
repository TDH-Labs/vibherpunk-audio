/* Vibherpunk Audio — shell cache only. API/WebRTC traffic is never cached.
   STANDING RULE: any change under client/ MUST bump CACHE below, or
   installed devices keep serving the stale shell (see RUNBOOK-mobile.md). */
const CACHE = "vibherpunk-audio-v7";
const SHELL = ["/", "/manifest.webmanifest", "/icon-192.png", "/icon-512.png"];

self.addEventListener("install", (ev) => {
  ev.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (ev) => {
  ev.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (ev) => {
  const url = new URL(ev.request.url);
  if (url.pathname.startsWith("/api/") || ev.request.method !== "GET") return;
  // Documents are network-first: a client fix must reach devices on the next
  // load, not whenever a cache entry happens to expire. The precached shell
  // is only the offline fallback. Static assets stay cache-first.
  if (ev.request.mode === "navigate") {
    ev.respondWith(
      fetch(ev.request).catch(() => caches.match(ev.request, { ignoreSearch: true }))
    );
    return;
  }
  ev.respondWith(
    caches.match(ev.request).then((hit) => hit || fetch(ev.request))
  );
});
