document.addEventListener("DOMContentLoaded", function () {
  var sel = "dl.py.attribute > dt .property, dl.py.data > dt .property";
  document.querySelectorAll(sel).forEach(function (prop) {
    var txt = prop.textContent || "";
    if (txt.length <= 80) return;                 // only long values
    prop.classList.add("long-value-collapsed");
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "long-value-toggle";
    btn.textContent = "show ▾";
    btn.addEventListener("click", function () {
      var collapsed = prop.classList.toggle("long-value-collapsed");
      btn.textContent = collapsed ? "show ▾" : "hide ▴";
    });
    prop.insertAdjacentElement("afterend", btn);
  });
});

// Accordion for the left-hand "On this page": click a top-level section to
// expand its whole subtree; click again (or another section) to collapse.
(function () {
  function init() {
    var toc = document.querySelector(".bd-sidebar-primary #pst-page-toc-nav");
    if (!toc) return;
    var tops = Array.prototype.slice.call(toc.querySelectorAll("li.toc-h2"));
    tops.forEach(function (li) {
      if (!li.querySelector("ul")) return;           // no children to toggle
      var a = li.querySelector(":scope > a");
      if (!a) return;
      a.addEventListener("click", function (e) {
        e.preventDefault();                          // act as a toggle, don't jump
        var wasOpen = li.classList.contains("expanded");
        tops.forEach(function (o) { o.classList.remove("expanded"); });  // accordion
        if (!wasOpen) li.classList.add("expanded");
      });
    });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
