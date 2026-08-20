/* Background-click channel for click-driven roof emphasis.
 *
 * Dash's dcc.Graph only updates clickData when plotly reports a point click
 * (plotly_click); clicking the plot background never fires it, so the roof
 * selection cannot be cleared by clicking empty space. This script tracks
 * plotly's own plotly_click events and, for DOM clicks inside the roofline
 * plot that are not point clicks, legend/modebar interactions, or hover
 * tooltips, bumps a hidden dcc.Input (roofline-bg-click). The server
 * callback reads that bump and clears the selection.
 *
 * The bump is deferred one tick so a point click (plotly_click, fired
 * synchronously during the same DOM event) is never mistaken for a
 * background click — plotly's draglayer overlay sits above the markers, so
 * the DOM target alone cannot distinguish the two. A boolean flag set by
 * plotly_click and consumed by the deferred check distinguishes them; the
 * flag is reset on every path so it cannot leak into a later click. The
 * input value is written through the native setter so React (Dash's
 * renderer) observes the change, and a monotonic counter is used because
 * Dash ignores input events whose value does not change.
 */
(function () {
    var PLOT_ID = "roofline-plot";
    var INPUT_ID = "roofline-bg-click";
    var pendingPointClick = false;
    var valueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;

    function isPointClick(target) {
        if (!target || !target.closest) {
            return false;
        }
        // .hoverlayer: tooltips have pointer-events:none but keep the guard anyway.
        return Boolean(target.closest(".point, .textpoint, .legend, .modebar, .hoverlayer"));
    }

    function attachPointClickWatch() {
        var plot = document.getElementById(PLOT_ID);
        if (!plot) {
            return;
        }
        var gd = plot.querySelector(".js-plotly-plot");
        if (gd && !gd.__carmPointClickWatch) {
            gd.__carmPointClickWatch = true;
            gd.on("plotly_click", function () {
                pendingPointClick = true;
            });
        }
    }

    function bumpBgClick(input) {
        var next = String((parseInt(input.value, 10) || 0) + 1);
        valueSetter.call(input, next);
        input.dispatchEvent(new Event("input", { bubbles: true }));
    }

    document.addEventListener("click", function (event) {
        var plot = document.getElementById(PLOT_ID);
        var input = document.getElementById(INPUT_ID);
        if (!plot || !input) {
            return;
        }
        attachPointClickWatch();
        if (!plot.contains(event.target)) {
            return;
        }
        if (isPointClick(event.target)) {
            // A point/legend/modebar/hover click never bumps; consume any flag
            // set by its plotly_click so a later background click is not skipped.
            pendingPointClick = false;
            return;
        }
        setTimeout(function () {
            var wasPointClick = pendingPointClick;
            pendingPointClick = false;
            if (!wasPointClick) {
                bumpBgClick(input);
            }
        }, 0);
    });

    // The layout is rendered by React after this asset script runs, so the
    // plot container does not exist yet. Watching document.body binds the
    // plotly_click watcher as soon as the plot mounts, before any user click.
    new MutationObserver(attachPointClickWatch).observe(document.body, { childList: true, subtree: true });
    attachPointClickWatch();
})();
