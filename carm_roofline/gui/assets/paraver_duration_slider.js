/* Tooltip formatter for the Paraver minimum-duration filter slider.
 *
 * Slider positions are log10(minimum duration in seconds); the tooltip shows
 * the threshold in human units (e.g. -4 -> "100 us", -3 -> "1 ms", -1 -> "100 ms").
 * The leftmost position (the slider's min, derived from DURATION_FILTER_OFF_S
 * in providers.py) disables filtering entirely. The off bound is read from the
 * rendered slider's min attribute rather than hardcoded, so the formatter can
 * never drift from the Python constants.
 */
window.dccFunctions = window.dccFunctions || {};
window.dccFunctions.paraverDuration = function (value) {
    var n = Number(value);
    if (!isFinite(n)) {
        return String(value);
    }
    var container = document.getElementById("slider-duration-threshold");
    var input = container ? container.querySelector("input") : null;
    var min = input ? Number(input.min) : NaN;
    if (!isNaN(min) && n <= min) {
        return "Off";
    }
    var s = Math.pow(10, n);
    if (s < 0.001) {
        return Math.round(s * 1e6) + " us";
    }
    if (s < 1) {
        return Math.round(s * 1e3) + " ms";
    }
    return Math.round(s) + " s";
};
