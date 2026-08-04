/* Tooltip formatter for the Paraver time-window slider.
 *
 * Dash's dcc.Slider/RangeSlider renders tooltip text from the raw slider value
 * (e.g. "0.26458415300000004") and exposes `tooltip.transform`: the name of a
 * function in the window.dccFunctions namespace that formats the value. The
 * slider passes `tooltip={"transform": "paraverTime"}`; this file registers the
 * function. Trailing zeros are stripped so 0.5 shows as "0.5", not "0.50000".
 */
window.dccFunctions = window.dccFunctions || {};
window.dccFunctions.paraverTime = function (value) {
    var n = Number(value);
    if (!isFinite(n)) {
        return String(value);
    }
    return n.toFixed(5).replace(/\.?0+$/, "");
};
