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
