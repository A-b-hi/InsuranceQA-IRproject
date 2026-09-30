/* Precision-Recall chart drawn from the real evaluation results (data/evaluation/results.json). */
(function () {
  "use strict";
  var data = JSON.parse(document.getElementById("eval-data").textContent);
  var order = ["tfidf", "bm25", "semantic", "hybrid"];
  var colors = { hybrid: "#2456E8", semantic: "#7A5AF8", bm25: "#0E9AA7", tfidf: "#E08A00" };
  var NS = "http://www.w3.org/2000/svg";
  var W = 760, H = 430, M = { l: 58, r: 22, t: 16, b: 54 };
  var iw = W - M.l - M.r, ih = H - M.t - M.b;
  var host = document.getElementById("pr-chart"), legend = document.getElementById("pr-legend");
  var hidden = {};

  var maxP = 0;
  order.forEach(function (m) { data.methods[m].pr_curve.forEach(function (p) { maxP = Math.max(maxP, p.precision); }); });
  var yMax = Math.min(1, Math.ceil(maxP * 10 + 0.5) / 10);
  var x = function (r) { return M.l + r * iw; };
  var y = function (p) { return M.t + ih - (p / yMax) * ih; };
  var el = function (name, attrs, parent) {
    var n = document.createElementNS(NS, name);
    Object.keys(attrs || {}).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    if (parent) parent.appendChild(n); return n;
  };

  var svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img",
    "aria-label": "Precision-Recall curves for TF-IDF, BM25, Semantic and Hybrid RRF" });
  host.appendChild(svg);

  // grid + axes
  for (var i = 0; i <= 10; i++) {
    var r = i / 10;
    el("line", { x1: x(r), x2: x(r), y1: M.t, y2: M.t + ih, class: "grid-line" }, svg);
    var t = el("text", { x: x(r), y: M.t + ih + 22, "text-anchor": "middle", class: "axis" }, svg); t.textContent = r.toFixed(1);
  }
  for (var p = 0; p <= yMax + 1e-9; p += 0.1) {
    el("line", { x1: M.l, x2: M.l + iw, y1: y(p), y2: y(p), class: "grid-line" }, svg);
    var yt = el("text", { x: M.l - 10, y: y(p) + 4, "text-anchor": "end", class: "axis" }, svg); yt.textContent = p.toFixed(1);
  }
  el("line", { x1: M.l, x2: M.l + iw, y1: M.t + ih, y2: M.t + ih, class: "axis-line" }, svg);
  var xl = el("text", { x: M.l + iw / 2, y: H - 10, "text-anchor": "middle", class: "axis" }, svg); xl.textContent = "Recall";
  var yl = el("text", { transform: "translate(16 " + (M.t + ih / 2) + ") rotate(-90)", "text-anchor": "middle", class: "axis" }, svg); yl.textContent = "Precision";

  // one group per method (hybrid drawn last, on top)
  var groups = {};
  order.forEach(function (m) {
    var g = el("g", {}, svg); groups[m] = g;
    var pts = data.methods[m].pr_curve;
    var d = pts.map(function (q, k) { return (k ? "L" : "M") + x(q.recall).toFixed(1) + " " + y(q.precision).toFixed(1); }).join(" ");
    el("path", { d: d, class: "pr-line " + m, stroke: colors[m] }, g);
    pts.forEach(function (q) { el("circle", { cx: x(q.recall), cy: y(q.precision), r: m === "hybrid" ? 4 : 3.2, fill: "#fff", stroke: colors[m], "stroke-width": 2 }, g); });
  });

  // legend toggles
  order.forEach(function (m) {
    var b = document.createElement("button"); b.type = "button"; b.className = "lg m-" + m;
    b.innerHTML = '<span class="dot" style="background:' + colors[m] + '"></span>' + data.methods[m].label;
    b.addEventListener("click", function () {
      hidden[m] = !hidden[m]; b.classList.toggle("off", hidden[m]);
      groups[m].style.display = hidden[m] ? "none" : "";
    });
    legend.appendChild(b);
  });

  // hover tooltip at the nearest recall level
  var guide = el("line", { y1: M.t, y2: M.t + ih, class: "guide", visibility: "hidden" }, svg);
  var tip = document.createElement("div"); tip.className = "tip"; tip.hidden = true; host.appendChild(tip);
  var overlay = el("rect", { x: M.l, y: M.t, width: iw, height: ih, fill: "transparent" }, svg);
  overlay.addEventListener("mousemove", function (e) {
    var box = svg.getBoundingClientRect(), scale = W / box.width;
    var r = Math.max(0, Math.min(1, ((e.clientX - box.left) * scale - M.l) / iw));
    var idx = Math.round(r * 10), lv = idx / 10;
    guide.setAttribute("x1", x(lv)); guide.setAttribute("x2", x(lv)); guide.setAttribute("visibility", "visible");
    var rows = order.filter(function (m) { return !hidden[m]; }).map(function (m) {
      return '<span><span style="color:' + colors[m] + '">&#9679; ' + data.methods[m].label + "</span><span>" + data.methods[m].pr_curve[idx].precision.toFixed(3) + "</span></span>";
    }).join("");
    tip.innerHTML = "<b>Recall " + lv.toFixed(1) + "</b>" + rows; tip.hidden = false;
    tip.style.left = (x(lv) / scale) + "px"; tip.style.top = ((M.t + 20) / scale) + "px";
  });
  overlay.addEventListener("mouseleave", function () { tip.hidden = true; guide.setAttribute("visibility", "hidden"); });
})();
