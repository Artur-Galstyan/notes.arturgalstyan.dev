// @ts-check
import { defineConfig } from "astro/config";

import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import tailwindcss from "@tailwindcss/vite";
import vercelStatic from "@astrojs/vercel";

// https://astro.build/config
export default defineConfig({
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
    shikiConfig: {
      themes: { light: "min-light", dark: "night-owl" },
    },
  },
  vite: {
    plugins: [tailwindcss()],
  },
  adapter: vercelStatic({
    webAnalytics: {
      enabled: true,
    },
  }),
  output: "static",
});
