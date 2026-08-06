/* Tooltip formatter for the Paraver arithmetic-intensity filter slider.
 *
 * Slider positions are log10(ai threshold); the tooltip shows the threshold
 * in scientific notation (e.g. -5 -> "1e-5", -4.5 -> "3e-5"). The leftmost
 * position (the slider's min, derived from AI_FILTER_OFF_AI in providers.py)
 * disables filtering entirely. The off bound is read from the rendered
 * slider's min attribute rather than hardcoded, so the formatter can never
 * drift from the Python constants.
 */
window.dccFunctions = window.dccFunctions || {};
window.dccFunctions.aiThreshold = function (value) {
    var n = Number(value);
    if (!isFinite(n)) {
        return String(value);
    }
    var container = document.getElementById("slider-ai-threshold");
    var input = container ? container.querySelector("input") : null;
    var min = input ? Number(input.min) : NaN;
    if (!isNaN(min) && n <= min) {
        return "Off";
    }
    return Math.pow(10, n).toExponential(0);
};
