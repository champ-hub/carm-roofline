/* Add full-name tooltips to legend entries.
 *
 * Python crops trace names before Plotly calculates the legend layout. The
 * complete name remains in trace metadata for this rendered SVG tooltip.
 */
(function () {
    var PLOT_ID = "roofline-plot";
    var FULL_LEGEND_NAME_META_KEY = "carm_full_legend_name";
    var SVG_NAMESPACE = "http://www.w3.org/2000/svg";

    function removeLegendTitles(label, traceGroup) {
        label.querySelectorAll("title").forEach(function (title) {
            title.remove();
        });
        if (traceGroup) {
            traceGroup.querySelectorAll("title[data-carm-legend-title]").forEach(function (title) {
                title.remove();
            });
        }
    }

    function fullLegendName(traceGroup) {
        var boundData = traceGroup && traceGroup.__data__;
        var trace = boundData && boundData[0] && boundData[0].trace;
        var meta = trace && trace.meta;
        return meta && typeof meta[FULL_LEGEND_NAME_META_KEY] === "string"
            ? meta[FULL_LEGEND_NAME_META_KEY]
            : null;
    }

    function formatLegendLabels(plot) {
        plot.querySelectorAll(".legend .legendtext").forEach(function (label) {
            var traceGroup = label.closest(".traces");
            removeLegendTitles(label, traceGroup);

            var visibleName = label.textContent;
            var fullName = fullLegendName(traceGroup);
            label.setAttribute("data-carm-display-name", visibleName);
            if (!fullName || fullName === visibleName || !traceGroup) {
                label.removeAttribute("data-carm-full-name");
                return;
            }

            label.setAttribute("data-carm-full-name", fullName);
            var title = document.createElementNS(SVG_NAMESPACE, "title");
            title.setAttribute("data-carm-legend-title", "");
            title.textContent = fullName;
            traceGroup.insertBefore(title, traceGroup.firstChild);
        });
    }

    function attachLegendFormatter() {
        var plotContainer = document.getElementById(PLOT_ID);
        if (!plotContainer) {
            return;
        }

        var plot = plotContainer.querySelector(".js-plotly-plot");
        if (plot && !plot.__carmLegendFormatter) {
            plot.__carmLegendFormatter = true;
            formatLegendLabels(plot);
            plot.on("plotly_afterplot", function () {
                formatLegendLabels(plot);
            });
        }
    }

    new MutationObserver(attachLegendFormatter).observe(document.body, { childList: true, subtree: true });
    attachLegendFormatter();
})();
