(() => {
    const overlayId = "folder-import-overlay";
    const statusId = "folder-import-status";
    const payloadId = "folder-import-payload";
    let active = false;
    let dragDepth = 0;

    const overlay = () => document.getElementById(overlayId);
    const show = () => {
        active = true;
        if (overlay()) overlay().classList.add("folder-import-overlay--visible");
    };
    const hide = () => {
        active = false;
        dragDepth = 0;
        if (overlay()) overlay().classList.remove("folder-import-overlay--visible");
    };
    const status = (message) => {
        if (window.dash_clientside && window.dash_clientside.set_props) {
            window.dash_clientside.set_props(statusId, { children: message });
        }
    };
    let statusTimer = null;
    const clearStatus = () => {
        window.clearTimeout(statusTimer);
        statusTimer = null;
        const node = document.getElementById(statusId);
        if (!node || !node.textContent.trim()) return;
        if (window.dash_clientside && window.dash_clientside.set_props) {
            window.dash_clientside.set_props(statusId, { children: null });
        }
    };
    const watchStatus = () => {
        const node = document.getElementById(statusId);
        if (!node) {
            window.setTimeout(watchStatus, 50);
            return;
        }
        const updateTimer = () => {
            window.clearTimeout(statusTimer);
            statusTimer = node.textContent.trim() ? window.setTimeout(clearStatus, 5000) : null;
        };
        new MutationObserver(updateTimer).observe(node, { childList: true, characterData: true, subtree: true });
        document.addEventListener("click", clearStatus, true);
        updateTimer();
    };
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", watchStatus, { once: true });
    } else {
        watchStatus();
    }
    const isFiles = (event) => Array.from(event.dataTransfer?.types || []).includes("Files");

    document.addEventListener("dragenter", (event) => {
        if (!isFiles(event)) return;
        dragDepth += 1;
        show();
    });
    document.addEventListener("dragover", (event) => {
        if (!isFiles(event)) return;
        event.preventDefault();
        event.dataTransfer.dropEffect = "copy";
        show();
    });
    document.addEventListener("dragleave", (event) => {
        if (!isFiles(event)) return;
        dragDepth = Math.max(0, dragDepth - 1);
        if (dragDepth === 0) hide();
    });
    document.addEventListener("dragend", hide);
    window.addEventListener("blur", hide);
    document.addEventListener("drop", async (event) => {
        if (!isFiles(event)) return;
        event.preventDefault();
        hide();
        const items = Array.from(event.dataTransfer.items || []).filter((item) => item.kind === "file");
        if (items.length !== 1 || typeof items[0].webkitGetAsEntry !== "function") {
            status("Import error: drop exactly one folder in a supported browser.");
            return;
        }
        const root = items[0].webkitGetAsEntry();
        if (!root || !root.isDirectory) {
            status("Import error: drop exactly one folder.");
            return;
        }
        const files = [];
        const readDirectory = (directory) => new Promise((resolve, reject) => {
            const reader = directory.createReader();
            const entries = [];
            const next = () => reader.readEntries((batch) => {
                if (!batch.length) return resolve(entries);
                entries.push(...batch);
                next();
            }, reject);
            next();
        });
        const walk = async (entry, relative) => {
            if (entry.isFile) {
                const depth = relative.split("/").length;
                if (depth > 2 || !["benchmarks.jsonl", "applications.jsonl"].includes(entry.name)) return;
                const file = await new Promise((resolve, reject) => entry.file(resolve, reject));
                files.push({ path: `${root.name}/${relative}`, content: await file.text() });
                return;
            }
            if (entry.isDirectory) {
                const children = await readDirectory(entry);
                for (const child of children) await walk(child, relative ? `${relative}/${child.name}` : child.name);
            }
        };
        try {
            await walk(root, "");
            window.dash_clientside.set_props(payloadId, { data: { files } });
        } catch (error) {
            status(`Import error: could not read the dropped folder (${error.message || error}).`);
        }
    });
    window.addEventListener("DOMContentLoaded", () => { if (!active) hide(); });
})();
