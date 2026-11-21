/*
 * Safari sizes the header row of a bokeh Tabs without the horizontal
 * scrollbar that appears when there are too many tabs for the available width.
 * The scrollbar then takes its space from the tabs, which end up squished until
 * the window is resized.  This happens some time after the tabs are rendered,
 * and again whenever bokeh re-renders them (e.g., when switching tabs).
 *
 * So, keep watching the tab headers for as long as the page lives, and make
 * the ones in that state taller by the thickness of their scrollbar.
 *
 * This is loaded with the page rather than hooked to the bokeh document-ready
 * event, since the latter can fire long after the tabs are first painted.  It
 * does not rely on requestAnimationFrame or on a time limit either, since a
 * page that is not visible gets no frames.  The observers react as soon as the
 * DOM or a size changes, and a slow interval catches anything they miss.
 *
 * The tabs are in the shadow DOM so they are only reachable from JavaScript.
 * This does nothing on a browser that gets the sizing right.
 */
(function() {
    function findTabsShadowRoots(root, found) {
        root.querySelectorAll("*").forEach(function(elem) {
            const sr = elem.shadowRoot;
            if (!sr)
                return;
            if (sr.querySelector(".bk-header") && sr.querySelector(".bk-tab"))
                found.push(sr);
            findTabsShadowRoots(sr, found);
        });
        return found;
    }

    function fixTabsHeader(sr) {
        const header = sr.querySelector(".bk-header");
        const wrapper = sr.querySelector(".bk-headers-wrapper");
        const tab = sr.querySelector(".bk-tab");
        if (!(header && wrapper && tab))
            return;
        const scrollbar = wrapper.offsetHeight - wrapper.clientHeight;
        if (scrollbar > 0 && tab.offsetHeight < tab.scrollHeight) {
            header.style.minHeight = (header.offsetHeight + scrollbar) + "px";
        } else if (scrollbar == 0 && header.style.minHeight) {
            // No more scrollbar (e.g., the window got wider): drop our fix
            header.style.minHeight = "";
        }
    }

    const watched = new WeakSet();

    function watchTabs() {
        findTabsShadowRoots(document, []).forEach(function(sr) {
            if (!watched.has(sr)) {
                watched.add(sr);
                const fix = function() { fixTabsHeader(sr); };
                new MutationObserver(fix).observe(
                    sr, {childList: true, subtree: true, attributes: true});
                const resizeObserver = new ResizeObserver(fix);
                resizeObserver.observe(sr.querySelector(".bk-header"));
                const wrapper = sr.querySelector(".bk-headers-wrapper");
                if (wrapper)
                    resizeObserver.observe(wrapper);
            }
            fixTabsHeader(sr);
        });
    }

    setInterval(watchTabs, 250);
})();
