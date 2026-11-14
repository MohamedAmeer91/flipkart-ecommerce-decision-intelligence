import { h } from "./api.js";
import * as pages from "./pages.js";

const ROUTES = [
  ["overview", "Overview", pages.overview], ["part1", "Part 1", pages.part1], ["part2", "Part 2", pages.part2],
  ["part3", "Part 3", pages.part3], ["part4", "Part 4", pages.part4], ["memo", "Memo", pages.memo],
  ["limitations", "Limitations", pages.limitations],
];
const main = document.getElementById("main");
const nav = document.getElementById("nav");
nav.append(...ROUTES.map(([id, label]) => h("a", { href: "#/" + id, "data-id": id }, label)));

async function render() {
  const id = (location.hash.replace(/^#\//, "") || "overview");
  const route = ROUTES.find((r) => r[0] === id) || ROUTES[0];
  nav.querySelectorAll("a").forEach((a) => (a.dataset.id === route[0] ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current")));
  document.title = route[1] + " | Flipkart Decision Intelligence";
  main.replaceChildren(h("p", { class: "muted" }, "Loading..."));
  try {
    main.replaceChildren(await route[2]());
  } catch (e) {
    main.replaceChildren(h("p", { class: "err", role: "alert" }, "Something went wrong: " + e.message));
  }
  window.scrollTo(0, 0);
}
window.addEventListener("hashchange", render);
render();
