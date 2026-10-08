/**
 * Result page: fetches the RF + LSTM hybrid analysis and fills the
 * "Random Forest" and "LSTM" cards plus the fused hybrid result.
 * All dynamic text is inserted with textContent (URLs are untrusted input).
 */
(function () {
    "use strict";
    var root = document.getElementById("hybrid-panel");
    if (!root) { return; }

    function el(tag, cls, text) {
        var node = document.createElement(tag);
        if (cls) { node.className = cls; }
        if (text !== undefined) { node.textContent = text; }
        return node;
    }
    function slot(id) { var n = document.getElementById(id); n.textContent = ""; return n; }
    function pct(p) { return (p * 100).toFixed(1) + "%"; }
    function pts(v) { return (v > 0 ? "+" : "") + v.toFixed(1) + " pts"; }
    function colour(p) { return p >= 0.7 ? "var(--phishing)" : p >= 0.35 ? "var(--suspicious)" : "var(--safe)"; }

    function probability(box, p, caption) {
        var big = el("div", "hy-prob", pct(p));
        big.style.color = colour(p);
        box.appendChild(big);
        box.appendChild(el("div", "small muted", caption));
        var bar = el("div", "hy-bar"), fill = document.createElement("div");
        fill.style.width = (p * 100).toFixed(1) + "%";
        fill.style.background = colour(p);
        bar.appendChild(fill);
        box.appendChild(bar);
    }

    function showError(message) {
        ["hy-fused", "hy-rf", "hy-lstm"].forEach(function (id) {
            slot(id).appendChild(el("p", "hy-error", message));
        });
    }

    function render(d) {
        // Fused hybrid result
        var f = slot("hy-fused");
        var main = el("div", "hy-fused__main");
        main.appendChild(el("div", "panel__title mb-0", "Final hybrid result"));
        var big = el("div", "hy-fused__big", pct(d.probability) + " phishing");
        big.style.color = colour(d.probability);
        main.appendChild(big);
        main.appendChild(el("div", "small", d.label + " \u00b7 " + d.verdict));
        f.appendChild(main);
        f.appendChild(el("div", "hy-fused__formula",
            pct(d.probability) + " = " + d.weights.rf + " \u00d7 RF (" + pct(d.rf_probability) +
            ") + " + d.weights.lstm + " \u00d7 LSTM (" + pct(d.lstm_probability) + ")"));

        // Random Forest card (+ SHAP)
        var rf = slot("hy-rf");
        probability(rf, d.rf_probability, "phishing probability \u00b7 handcrafted URL features");
        var shap = d.shap;
        if (shap && shap.available) {
            rf.appendChild(el("div", "hy-note", "SHAP: what moved this score (probability points). " +
                "Explains this branch only."));
            var max = Math.max.apply(null, shap.contributions.map(function (c) { return Math.abs(c.shap); }).concat([1]));
            shap.contributions.forEach(function (c) {
                var row = el("div", "factor");
                var name = el("div", "factor__name");
                name.appendChild(el("div", "factor__label", c.label));
                name.appendChild(el("div", "factor__value", "value: " + c.value));
                var bar = el("div", "factor__bar");
                var fill = el("div", "factor__fill " + (c.shap > 0 ? "factor__fill--up" : "factor__fill--down"));
                fill.style.width = Math.min(Math.abs(c.shap) / max * 100, 100) + "%";
                bar.appendChild(fill);
                row.appendChild(name);
                row.appendChild(bar);
                row.appendChild(el("div", "factor__impact " + (c.shap > 0 ? "impact-up" : "impact-down"), pts(c.shap)));
                rf.appendChild(row);
            });
        } else {
            rf.appendChild(el("p", "hy-note", shap ? shap.message : "SHAP unavailable."));
        }

        // LSTM card (+ character occlusion)
        var ls = slot("hy-lstm");
        probability(ls, d.lstm_probability, "phishing probability \u00b7 raw URL characters");
        var att = d.lstm_attribution;
        if (att) {
            ls.appendChild(el("div", "hy-note", "Character influence (occlusion, not SHAP): red pushed the LSTM " +
                "towards phishing, green towards legitimate. Shown as the LSTM sees the URL (scheme removed, lower-case)."));
            var strip = el("div", "hy-chars");
            var maxAbs = Math.max.apply(null, att.characters.map(function (c) { return Math.abs(c.impact); }).concat([0.5]));
            att.characters.forEach(function (c) {
                var span = el("span", "", c.char);
                var a = Math.min(Math.abs(c.impact) / maxAbs, 1) * 0.85;
                span.style.background = (c.impact >= 0 ? "rgba(251,113,133," : "rgba(52,211,153,") + a.toFixed(2) + ")";
                span.title = pts(c.impact);
                strip.appendChild(span);
            });
            ls.appendChild(strip);
        }
    }

    fetch(root.getAttribute("data-endpoint") + "?explain=1", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: root.getAttribute("data-url") })
    }).then(function (r) {
        return r.json().then(function (j) { return { ok: r.ok, json: j }; });
    }).then(function (res) {
        if (res.ok) { render(res.json); }
        else { showError("Unavailable: " + (res.json.message || res.json.error || "unknown error")); }
    }).catch(function () {
        showError("Could not load the hybrid analysis (see the browser console).");
    });
}());
