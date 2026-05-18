import { useEffect, useRef } from 'react';

export function MapBackground() {
  const mapRef = useRef<HTMLDivElement>(null);
  const leafletMap = useRef<any>(null);

  useEffect(() => {
    if (!mapRef.current || leafletMap.current) return;

    const initMap = () => {
      const L = window.L;
      if (!L) {
        setTimeout(initMap, 100);
        return;
      }

      const map = L.map(mapRef.current, {
        center: [37.7749, -122.4194], // San Francisco
        zoom: 13,
        zoomControl: false,
        attributionControl: false,
        dragging: false,
        scrollWheelZoom: false,
        doubleClickZoom: false,
        keyboard: false,
        touchZoom: false
      });

      L.tileLayer(
        'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
        { subdomains: 'abcd', maxZoom: 19 }
      ).addTo(map);

      leafletMap.current = map;
    };

    if (!document.getElementById('leaflet-script')) {
      const script = document.createElement('script');
      script.id = 'leaflet-script';
      script.src = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';
      script.onload = initMap;
      document.head.appendChild(script);
    } else {
      initMap();
    }
  }, []);

  useEffect(() => {
    const handleFlyTo = (e: any) => {
      if (leafletMap.current) {
        leafletMap.current.flyTo([e.detail.lat, e.detail.lng], 13, { duration: 1.5 });
      }
    };
    window.addEventListener('FlyToCity', handleFlyTo);
    return () => window.removeEventListener('FlyToCity', handleFlyTo);
  }, []);

  return (
    <div style={{ position: 'fixed', inset: 0, zIndex: -10 }}>
      {/* The actual leaflet container */}
      <div ref={mapRef} style={{ width: '100%', height: '100%', opacity: 1 }} />
      {/* Very faint dark overlay so the original dark_all map remains bright and clearly visible */}
      <div style={{ position: 'absolute', inset: 0, background: 'rgba(6, 10, 20, 0.20)', pointerEvents: 'none' }} />
    </div>
  );
}
