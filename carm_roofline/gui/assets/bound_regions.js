/* Keep bound-region fills and labels clipped and centered in the visible roof plot. */
(function () {
    var PLOT_ID = "roofline-plot";
    var META_KEY = "carm_bound_region_specs";
    var ROLE = "bound-region";
    var ANNOTATION_PREFIX = "carm-bound-region:";

    function updateBoundRegions(plot) {
        var specs = plot.layout && plot.layout.meta && plot.layout.meta[META_KEY];
        var axis = plot._fullLayout && plot._fullLayout.xaxis;
        var yaxis = plot._fullLayout && plot._fullLayout.yaxis;
        if (!specs || !axis || !yaxis || !Array.isArray(axis.range) || !Array.isArray(yaxis.range)) {
            return;
        }

        var xMin = Math.pow(10, Math.min(axis.range[0], axis.range[1]));
        var xMax = Math.pow(10, Math.max(axis.range[0], axis.range[1]));
        var baseline = Math.pow(10, Math.min(yaxis.range[0], yaxis.range[1]));
        var maxY = Math.pow(10, Math.max(yaxis.range[0], yaxis.range[1]));
        var roofLines = specs.roof_lines || [];
        var traceIndices = [];
        var traceXs = [];
        var traceYs = [];
        var annotationUpdates = {};

        (plot.data || []).forEach(function (trace, index) {
            if (!trace.meta || trace.meta.carm_role !== ROLE) {
                return;
            }
            var label = trace.meta.bound_region;
            var region = (specs.regions || []).find(function (item) { return item.label === label; });
            if (!region) {
                return;
            }
            var left = region.min == null ? xMin : Math.max(xMin, region.min);
            var right = region.max == null ? xMax : Math.min(xMax, region.max);
            var xs = [];
            var ys = [];
            var visible = right > left;
            var indexOfAnnotation = annotationIndex(label, plot.layout.annotations);
            if (visible) {
                var knots = [left, right];
                roofLines.forEach(function (line) {
                    var slope = line[0];
                    [baseline, maxY].forEach(function (limit) {
                        var crossing = limit / slope;
                        if (crossing > left && crossing < right) {
                            knots.push(crossing);
                        }
                    });
                    roofLines.forEach(function (other) {
                        var crossing = other[1] / slope;
                        if (crossing > left && crossing < right) {
                            knots.push(crossing);
                        }
                    });
                });
                knots.sort(function (a, b) { return a - b; });
                knots = knots.filter(function (value, position) {
                    return position === 0 || value !== knots[position - 1];
                });
                var top = knots.map(function (intensity) {
                    var envelope = roofLines.reduce(function (maximum, line) {
                        return Math.max(maximum, Math.min(line[0] * intensity, line[1]));
                    }, 0);
                    return Math.max(baseline, Math.min(envelope, maxY));
                });
                xs = knots.concat(knots.slice().reverse());
                ys = top.concat(knots.map(function () { return baseline; }));
                var center = (Math.log10(left) + Math.log10(right)) / 2;
                if (
                    indexOfAnnotation >= 0 &&
                    Math.abs(plot.layout.annotations[indexOfAnnotation].x - center) > 1e-9
                ) {
                    annotationUpdates["annotations[" + indexOfAnnotation + "].x"] = center;
                }
            }
            if (
                indexOfAnnotation >= 0 &&
                plot.layout.annotations[indexOfAnnotation].visible !== visible
            ) {
                annotationUpdates["annotations[" + indexOfAnnotation + "].visible"] = visible;
            }
            if (!arraysEqual(trace.x || [], xs) || !arraysEqual(trace.y || [], ys)) {
                traceIndices.push(index);
                traceXs.push(xs);
                traceYs.push(ys);
            }
        });

        if (traceIndices.length) {
            window.Plotly.restyle(plot, { x: traceXs, y: traceYs }, traceIndices);
        }
        if (Object.keys(annotationUpdates).length) {
            window.Plotly.relayout(plot, annotationUpdates);
        }
    }

    function arraysEqual(left, right) {
        return left.length === right.length && left.every(function (value, index) {
            return value === right[index];
        });
    }

    function annotationIndex(label, annotations) {
        var name = ANNOTATION_PREFIX + label;
        return (annotations || []).findIndex(function (annotation) { return annotation.name === name; });
    }

    function attach() {
        var container = document.getElementById(PLOT_ID);
        var plot = container && container.querySelector(".js-plotly-plot");
        if (!plot || plot.__carmBoundRegionUpdater) {
            return;
        }
        plot.__carmBoundRegionUpdater = true;
        plot.on("plotly_afterplot", function () { updateBoundRegions(plot); });
        plot.on("plotly_relayout", function (eventData) {
            if (Object.keys(eventData || {}).some(function (key) {
                return key.indexOf("xaxis.range") === 0 || key.indexOf("yaxis.range") === 0 ||
                    key === "xaxis.autorange" || key === "yaxis.autorange";
            })) {
                updateBoundRegions(plot);
            }
        });
    }

    new MutationObserver(attach).observe(document.body, { childList: true, subtree: true });
    attach();
})();
