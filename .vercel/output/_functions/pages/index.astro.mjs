import { $ as $$Index$1, a as $$Navbar, _ as __vite_glob_0_0 } from '../chunks/mb-rl-experiments_Dvih2ihi.mjs';
import { c as createComponent, a as createAstro, b as addAttribute, r as renderComponent, d as renderHead, e as renderSlot, f as renderTemplate, m as maybeRenderHead } from '../chunks/astro/server_jCnFmPOC.mjs';
import 'kleur/colors';
/* empty css                                 */
export { renderers } from '../renderers.mjs';

const $$Astro = createAstro();
const $$Layout = createComponent(($$result, $$props, $$slots) => {
  const Astro2 = $$result.createAstro($$Astro, $$props, $$slots);
  Astro2.self = $$Layout;
  return renderTemplate`<html lang="en" data-theme="light"> <head><meta charset="utf-8"><link rel="icon" type="image/svg+xml" href="/favicon.svg"><meta name="viewport" content="width=device-width"><meta name="generator"${addAttribute(Astro2.generator, "content")}>${renderComponent($$result, "Analytics", $$Index$1, {})}${renderHead()}</head> <body class="bg-base-100 text-base-content"> <div class="container mx-auto px-4 py-8"> ${renderComponent($$result, "Navbar", $$Navbar, {})} </div> <div class="container mx-auto w-2/3"> ${renderSlot($$result, $$slots["default"])} </div> </body></html>`;
}, "/Users/arturgalstyan/Workspace/notes.arturgalstyan.dev/src/layouts/Layout.astro", void 0);

const $$Index = createComponent(($$result, $$props, $$slots) => {
  const posts = Object.values([__vite_glob_0_0]);
  return renderTemplate`${renderComponent($$result, "Layout", $$Layout, {}, { "default": ($$result2) => renderTemplate` ${maybeRenderHead()}<ul class="list-disc"> ${posts.map((post) => renderTemplate`<li class="mb-4"> <a class="text-blue-500 hover:text-blue-700"${addAttribute(post.url, "href")}> ${post.frontmatter.title} </a>${" "} <span class="text-gray-500"> ${new Date(post.frontmatter.date).toLocaleDateString()} </span> </li>`)} </ul> ` })}`;
}, "/Users/arturgalstyan/Workspace/notes.arturgalstyan.dev/src/pages/index.astro", void 0);

const $$file = "/Users/arturgalstyan/Workspace/notes.arturgalstyan.dev/src/pages/index.astro";
const $$url = "";

const _page = /*#__PURE__*/Object.freeze(/*#__PURE__*/Object.defineProperty({
    __proto__: null,
    default: $$Index,
    file: $$file,
    url: $$url
}, Symbol.toStringTag, { value: 'Module' }));

const page = () => _page;

export { page };
