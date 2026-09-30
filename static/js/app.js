/* PolicyLens interactions: loading state, example chips, expandable answers, copy, compare highlight. */
(function () {
  "use strict";
  var body = document.body;

  // Loading state while a search is running (progress bar, spinner, skeleton cards).
  document.querySelectorAll("form[data-loading]").forEach(function (form) {
    form.addEventListener("submit", function () { body.classList.add("is-searching"); });
  });
  window.addEventListener("pageshow", function (e) { if (e.persisted) body.classList.remove("is-searching"); });

  // Example chips run a real dataset question.
  document.querySelectorAll("[data-example]").forEach(function (chip) {
    chip.addEventListener("click", function () {
      var form = chip.closest("form");
      form.querySelector("input[name=q]").value = chip.dataset.example;
      form.requestSubmit();
    });
  });

  // Compare switch dims the single-method selector.
  document.querySelectorAll(".search-form").forEach(function (form) {
    var sw = form.querySelector(".switch input"), methods = form.querySelector(".methods");
    if (!sw || !methods) return;
    var sync = function () { methods.classList.toggle("dim", sw.checked); };
    sw.addEventListener("change", sync); sync();
  });

  // Expand / collapse long answers (toggle only appears when the text is actually clipped).
  function setupAnswers() {
    document.querySelectorAll(".answer").forEach(function (card) {
      var text = card.querySelector("[data-clamp]"), more = card.querySelector("[data-more]");
      if (!text || !more) return;
      if (text.scrollHeight > text.clientHeight + 2) more.hidden = false;
      more.addEventListener("click", function () {
        var open = text.classList.toggle("expanded");
        more.setAttribute("aria-expanded", open ? "true" : "false");
        more.querySelector("span").textContent = open ? "Show less" : "Show full answer";
      });
    });
  }
  (document.fonts && document.fonts.ready ? document.fonts.ready : Promise.resolve()).then(setupAnswers);

  // Copy answer text.
  document.querySelectorAll("[data-copy]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var text = btn.closest(".answer").querySelector("[data-clamp]").textContent.trim();
      var label = btn.querySelector("span");
      var done = function () { label.textContent = "Copied"; setTimeout(function () { label.textContent = "Copy answer"; }, 1600); };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, done);
      } else {
        var ta = document.createElement("textarea"); ta.value = text; document.body.appendChild(ta);
        ta.select(); try { document.execCommand("copy"); } catch (e) {} ta.remove(); done();
      }
    });
  });

  // Compare view: hovering an answer highlights the same answer in the other lists.
  var cards = document.querySelectorAll(".compare-grid .answer");
  cards.forEach(function (card) {
    var set = function (on) {
      document.querySelectorAll('.compare-grid .answer[data-doc="' + card.dataset.doc + '"]')
        .forEach(function (c) { c.classList.toggle("is-linked", on); });
    };
    card.addEventListener("mouseenter", function () { set(true); });
    card.addEventListener("mouseleave", function () { set(false); });
  });
})();
