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


    var LABEL_OVERLAY_CLASS = "carm-roof-label-overlay";

    function clipSegment(x0, y0, x1, y1, bounds) {
        var dx = x1 - x0;
        var dy = y1 - y0;
        var t0 = 0;
        var t1 = 1;
        var edges = [
            [-dx, x0 - bounds.xMin],
            [dx, bounds.xMax - x0],
            [-dy, y0 - bounds.yMin],
            [dy, bounds.yMax - y0]
        ];
        for (var i = 0; i < edges.length; i += 1) {
            var p = edges[i][0];
            var q = edges[i][1];
            if (p === 0) {
                if (q < 0) { return null; }
            } else {
                var t = q / p;
                if (p < 0) {
                    if (t > t1) { return null; }
                    t0 = Math.max(t0, t);
                } else {
                    if (t < t0) { return null; }
                    t1 = Math.min(t1, t);
                }
            }
        }
        return [x0 + t0 * dx, y0 + t0 * dy, x0 + t1 * dx, y0 + t1 * dy];
    }

    function updateRoofLabels(plot) {
        var fullLayout = plot._fullLayout;
        var xaxis = fullLayout && fullLayout.xaxis;
        var yaxis = fullLayout && fullLayout.yaxis;
        var size = fullLayout && fullLayout._size;
        if (!xaxis || !yaxis || !size || !Array.isArray(xaxis.range) || !Array.isArray(yaxis.range)) {
            return;
        }
        var overlay = plot.querySelector("." + LABEL_OVERLAY_CLASS);
        if (plot.layout.meta && plot.layout.meta.carm_show_roof_labels === false) {
            if (overlay) {
                overlay.style.display = "none";
            }
            return;
        }
        if (!overlay) {
            overlay = document.createElement("div");
            overlay.className = LABEL_OVERLAY_CLASS;
            overlay.style.cssText = "position:absolute;inset:0;overflow:hidden;pointer-events:none;z-index:20;";
            plot.style.position = "relative";
            plot.appendChild(overlay);
        }
        overlay.style.display = "block";
        var bounds = {
            xMin: Math.min(xaxis.range[0], xaxis.range[1]),
            xMax: Math.max(xaxis.range[0], xaxis.range[1]),
            yMin: Math.min(yaxis.range[0], yaxis.range[1]),
            yMax: Math.max(yaxis.range[0], yaxis.range[1])
        };
        var liveLabels = {};
        var context = document.createElement("canvas").getContext("2d");
        if (context) {
            context.font = "600 11px Arial,sans-serif";
        }
        (plot.data || []).forEach(function (trace, index) {
            if (!trace.meta || trace.meta.carm_role !== "roof-ceiling") {
                return;
            }
            var key = "roof-label-" + index;
            liveLabels[key] = true;
            var label = overlay.querySelector('[data-label-id="' + key + '"]');
            if (!label) {
                label = document.createElement("div");
                label.dataset.labelId = key;
                label.style.cssText =
                    "position:absolute;white-space:nowrap;transform-origin:center center;" +
                    "color:#fff;padding:2px 4px;font:600 11px Arial,sans-serif;" +
                    "border-radius:0;line-height:14px;";
                overlay.appendChild(label);
            }
            label.style.backgroundColor = trace.line.color;
            label.textContent = trace.meta.carm_roof_label;

            var visible = trace.visible !== false && trace.visible !== "legendonly" &&
                trace.opacity !== 0 && trace.x && trace.y && trace.x.length >= 2 && trace.y.length >= 2;
            var segment = visible ? clipSegment(
                Math.log10(trace.x[0]), Math.log10(trace.y[0]),
                Math.log10(trace.x[1]), Math.log10(trace.y[1]), bounds
            ) : null;
            if (!segment) {
                label.style.display = "none";
                return;
            }
            var px0 = size.l + xaxis.d2p(Math.pow(10, segment[0]));
            var py0 = size.t + yaxis.d2p(Math.pow(10, segment[1]));
            var px1 = size.l + xaxis.d2p(Math.pow(10, segment[2]));
            var py1 = size.t + yaxis.d2p(Math.pow(10, segment[3]));
            var dx = px1 - px0;
            var dy = py1 - py0;
            var length = Math.sqrt(dx * dx + dy * dy);
            if (!length) {
                label.style.display = "none";
                return;
            }
            var angle = Math.atan2(dy, dx) * 180 / Math.PI;
            var measuredWidth = context ? context.measureText(trace.meta.carm_roof_label).width + 8 : 24;
            var halfWidth = Math.max(measuredWidth, 20) / 2;
            var inset = Math.min(halfWidth + 3, length / 2);
            var atRight = trace.meta.carm_roof_label !== "L1" &&
                trace.meta.carm_roof_label !== "L2" &&
                trace.meta.carm_roof_label !== "L3" &&
                trace.meta.carm_roof_label !== "DRAM";
            var distance = atRight ? length - inset : inset;
            var x = px0 + dx / length * distance;
            var y = py0 + dy / length * distance;
            var radians = angle * Math.PI / 180;
            var normalX = Math.sin(radians);
            var normalY = -Math.cos(radians);
            var halfHeight = 11;
            x += normalX * (halfHeight + 2);
            y += normalY * (halfHeight + 2);
            var halfBoxX = Math.abs(Math.cos(radians)) * halfWidth +
                Math.abs(Math.sin(radians)) * halfHeight;
            var halfBoxY = Math.abs(Math.sin(radians)) * halfWidth +
                Math.abs(Math.cos(radians)) * halfHeight;
            x = Math.max(size.l + halfBoxX, Math.min(size.l + size.w - halfBoxX, x));
            y = Math.max(size.t + halfBoxY, Math.min(size.t + size.h - halfBoxY, y));
            label.style.display = "block";
            label.style.left = x + "px";
            label.style.top = y + "px";
            label.style.transform = "translate(-50%,-50%) rotate(" + angle + "deg)";
        });
        Array.prototype.slice.call(overlay.children).forEach(function (label) {
            if (!liveLabels[label.dataset.labelId]) {
                label.remove();
            }
        });
    }

    function attach() {
        var container = document.getElementById(PLOT_ID);
        var plot = container && container.querySelector(".js-plotly-plot");
        if (!plot || plot.__carmBoundRegionUpdater) {
            return;
        }
        plot.__carmBoundRegionUpdater = true;
        plot.on("plotly_afterplot", function () {
            updateBoundRegions(plot);
            updateRoofLabels(plot);
        });
        updateBoundRegions(plot);
        updateRoofLabels(plot);
        plot.on("plotly_relayout", function (eventData) {
            if (Object.keys(eventData || {}).some(function (key) {
                return key.indexOf("xaxis.range") === 0 || key.indexOf("yaxis.range") === 0 ||
                    key === "xaxis.autorange" || key === "yaxis.autorange";
            })) {
                updateBoundRegions(plot);
                updateRoofLabels(plot);
            }
        });
    }

    new MutationObserver(attach).observe(document.body, { childList: true, subtree: true });
    attach();
})();
