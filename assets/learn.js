(function () {
  "use strict";

  const STORE_KEY = "lfbs:v1";
  const MIN_HEADINGS_FOR_TOC = 4;

  function readState() {
    try {
      const parsed = JSON.parse(localStorage.getItem(STORE_KEY) || "{}");
      return { done: parsed.done || {}, last: parsed.last || {}, checks: parsed.checks || {} };
    } catch (err) {
      return { done: {}, last: {}, checks: {} };
    }
  }

  function writeState(state) {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify(state));
    } catch (err) {
      console.warn("Progress could not be saved in this browser:", err);
    }
  }

  function meta(name) {
    const tag = document.querySelector('meta[name="lfbs-' + name + '"]');
    return tag ? tag.content : null;
  }

  const state = readState();
  const pagePath = meta("path");
  const rootUrl = new URL(meta("root") || "./", location.href);
  const manifest = window.LFBS || { names: {}, lessons: {} };

  function sitePath(href) {
    let url;
    try {
      url = new URL(href, location.href);
    } catch (err) {
      return null;
    }
    if (url.origin !== rootUrl.origin || !url.pathname.startsWith(rootUrl.pathname)) return null;
    let path = decodeURIComponent(url.pathname.slice(rootUrl.pathname.length));
    if (path === "" || path.endsWith("/")) path += "index.html";
    return path;
  }

  function siteHref(path) {
    return new URL(path, rootUrl).href;
  }

  function el(tag, attrs, text) {
    const node = document.createElement(tag);
    Object.entries(attrs || {}).forEach(([key, value]) => node.setAttribute(key, value));
    if (text) node.textContent = text;
    return node;
  }

  function markDoneLinks() {
    document.querySelectorAll("a[href]").forEach((link) => {
      const path = sitePath(link.getAttribute("href"));
      link.classList.toggle("lfbs-done-link", Boolean(path && path !== pagePath && state.done[path]));
    });
  }

  function setDone(path, isDone) {
    if (isDone) state.done[path] = Date.now();
    else delete state.done[path];
    writeState(state);
  }

  function addDoneButton(content) {
    const bar = el("div", { class: "lfbs-done-bar" });
    const button = el("button", { type: "button", class: "lfbs-done-btn" });
    const note = el("span", { class: "lfbs-done-note" }, "Saved in this browser only.");
    const render = () => {
      const isDone = Boolean(state.done[pagePath]);
      button.setAttribute("aria-pressed", String(isDone));
      button.textContent = isDone ? "✓ Done" : "Mark as done";
    };
    button.addEventListener("click", () => {
      setDone(pagePath, !state.done[pagePath]);
      render();
    });
    render();
    bar.append(button, note);
    content.appendChild(bar);
  }

  function rememberVisit() {
    const track = meta("track");
    if (!track) return;
    state.last[track] = { path: pagePath, title: document.title.split(" - ")[0], at: Date.now() };
    writeState(state);
  }

  function upgradeChecklists(content) {
    const items = Array.from(content.querySelectorAll("li")).filter((li) =>
      /^\s*\[( |x|X)\]\s/.test(li.firstChild && li.firstChild.nodeType === 3 ? li.firstChild.nodeValue : ""));
    items.forEach((li, index) => {
      const textNode = li.firstChild;
      const initiallyTicked = /\[(x|X)\]/.test(textNode.nodeValue);
      textNode.nodeValue = textNode.nodeValue.replace(/^\s*\[( |x|X)\]\s/, "");
      const link = li.querySelector("a[href]");
      const lessonPath = link && sitePath(link.getAttribute("href"));
      const key = pagePath + "#" + index;
      const box = el("input", { type: "checkbox", class: "lfbs-check" });
      box.checked = lessonPath ? Boolean(state.done[lessonPath]) : key in state.checks ? state.checks[key] : initiallyTicked;
      box.addEventListener("change", () => {
        if (lessonPath) setDone(lessonPath, box.checked);
        else {
          state.checks[key] = box.checked;
          writeState(state);
        }
        markDoneLinks();
      });
      li.classList.add("lfbs-task");
      li.insertBefore(box, li.firstChild);
    });
  }

  function buildTocList(headings) {
    const list = el("ol");
    headings.forEach((heading) => {
      const item = el("li", heading.tagName === "H3" ? { class: "lfbs-toc-sub" } : {});
      item.appendChild(el("a", { href: "#" + heading.id }, heading.dataset.plain || heading.textContent));
      list.appendChild(item);
    });
    return list;
  }

  function addTableOfContents(content) {
    const headings = Array.from(content.querySelectorAll("h2[id], h3[id]")).filter((h) => !h.closest("details"));
    if (headings.filter((h) => h.tagName === "H2").length < MIN_HEADINGS_FOR_TOC) return;
    const inline = el("details", { class: "lfbs-toc-inline" });
    inline.appendChild(el("summary", {}, "On this page"));
    inline.appendChild(buildTocList(headings));
    const title = content.querySelector("h1");
    content.insertBefore(inline, title ? title.nextSibling : content.firstChild);

    const side = el("nav", { class: "lfbs-toc-side", "aria-label": "On this page" });
    side.appendChild(el("div", { class: "lfbs-toc-title" }, "On this page"));
    side.appendChild(buildTocList(headings.filter((h) => h.tagName === "H2")));
    document.body.appendChild(side);
    document.body.classList.add("lfbs-has-side-toc");

    const links = new Map(Array.from(side.querySelectorAll("a")).map((a) => [a.getAttribute("href").slice(1), a]));
    const observer = new IntersectionObserver((entries) => {
      entries.filter((e) => e.isIntersecting).forEach((entry) => {
        links.forEach((a) => a.classList.remove("lfbs-active"));
        const active = links.get(entry.target.id);
        if (active) active.classList.add("lfbs-active");
      });
    }, { rootMargin: "0px 0px -70% 0px" });
    headings.filter((h) => h.tagName === "H2").forEach((h) => observer.observe(h));
  }

  function progressRow(trackKey) {
    const lessons = manifest.lessons[trackKey] || [];
    if (!lessons.length) return null;
    const doneCount = lessons.filter(([path]) => state.done[path]).length;
    const last = state.last[trackKey];
    const nextLesson = lessons.find(([path]) => !state.done[path]);
    const row = el("div", { class: "lfbs-progress-row" });
    row.appendChild(el("span", { class: "lfbs-progress-name" }, manifest.names[trackKey] || trackKey));
    const bar = el("div", { class: "lfbs-progress-track", role: "progressbar", "aria-valuemin": "0",
      "aria-valuemax": String(lessons.length), "aria-valuenow": String(doneCount) });
    bar.appendChild(el("div", { class: "lfbs-progress-fill", style: "width:" + (100 * doneCount / lessons.length).toFixed(1) + "%" }));
    row.appendChild(bar);
    row.appendChild(el("span", { class: "lfbs-progress-count" }, doneCount + " / " + lessons.length));
    const target = last ? [last.path, "Continue: " + last.title] : nextLesson ? [nextLesson[0], "Start: " + nextLesson[1]] : null;
    if (target) row.appendChild(el("a", { class: "lfbs-continue", href: siteHref(target[0]) }, target[1] + " →"));
    return { row, started: doneCount > 0 || Boolean(last) };
  }

  function addProgressCard(trackKeys, heading, onlyStarted) {
    const rows = trackKeys.map(progressRow).filter((r) => r && (!onlyStarted || r.started));
    if (!rows.length) return;
    const card = el("section", { class: "lfbs-progress-card", "aria-label": heading });
    card.appendChild(el("h2", {}, heading));
    rows.forEach((r) => card.appendChild(r.row));
    const slot = document.getElementById("lfbs-progress");
    if (slot) slot.appendChild(card);
    else {
      const main = document.querySelector("main") || document.querySelector(".container") || document.body;
      main.insertBefore(card, main.firstElementChild ? main.firstElementChild.nextSibling : null);
    }
  }

  function init() {
    const content = document.querySelector(".md-content");
    if (content) {
      upgradeChecklists(content);
      addTableOfContents(content);
      if (meta("lesson") === "1") addDoneButton(content);
    }
    if (meta("lesson") === "1") rememberVisit();
    const landing = meta("landing");
    if (landing) addProgressCard(landing.split(","), "Your progress", false);
    if (meta("home") === "1") addProgressCard(Object.keys(manifest.lessons), "Continue learning", true);
    markDoneLinks();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
