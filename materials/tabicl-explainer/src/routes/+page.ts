// The upstream app uses browser-only APIs during module initialization.
// Keep static output, but render the visualization only in the browser.
export const prerender = true;
export const ssr = false;
