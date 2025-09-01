(function () {
    var infobox_stick = null;
    var infobox_source_el = null;
    const infobox_items = {};
    const infobox = document.querySelectorAll('#ID .pts-infobox')[0];
    const infobox_title = document.querySelectorAll('#ID .pts-infobox-title')[0];
    const infobox_body = document.querySelectorAll('#ID .pts-infobox-body')[0];
    const infobox_hint =  document.querySelectorAll('#ID .pts-infobox-hint')[0];
    const container = document.querySelectorAll('#ID')[0];
    function replace_infobox(el) {
        if (infobox_source_el !== null) {
            infobox_source_el.classList.remove('pts-infobox-source');
            infobox_source_el = null;
        }
        infobox_body.innerHTML = '';
        // use camelcase to access dataset attributes with - in the name 
        infobox_title.textContent = infobox_items[el.dataset.ptsInfobox].dataset.title || '';
        infobox.style.display = 'block';
        infobox_body.appendChild(infobox_items[el.dataset.ptsInfobox]);
        if (infobox_body.querySelectorAll('.pts-infobox-error').length > 0) {
            infobox.classList.add('pts-infobox-outer-error');
        } else {
            infobox.classList.remove('pts-infobox-outer-error');
        }
        infobox.scrollTop = 0;
        infobox_body.scrollLeft = 0;
        const infRect = infobox.getBoundingClientRect();
        const contRect = container.getBoundingClientRect();
        const elRect = el.getBoundingClientRect();
        if (elRect.left - contRect.left > infRect.width + 14) {
        // move the infobox to the immediate left/right of this element, depending on where there is space
            infobox.style.left = (elRect.left - contRect.left - infRect.width - 10) + 'px';
        } else {
            infobox.style.left = (elRect.right - contRect.left + 2) + 'px';
        }
        // Move to top of this element (if there is vertical space, otherwise as close as possible)
        var top = elRect.top - contRect.top;
        if (top + infRect.height > contRect.height) {
            top = contRect.height - infRect.height;
        }
        infobox.style.top = top + 'px';
        infobox_source_el = el;
        el.classList.add('pts-infobox-source');
    }
    function hide_infobox() {
        if (infobox_source_el !== null) {
            infobox_source_el.classList.remove('pts-infobox-source');
            infobox_source_el = null;
        }
        infobox_stick = null;
        infobox.style.display = 'none';
        infobox.style.opacity = '';
    }
    container.addEventListener('click', () => {
        if (infobox_stick) {
            hide_infobox();
        }
    });
    document.querySelectorAll('#ID .pts-infobox-item').forEach(el => {
        el.remove();
        el.style.display = 'block';
        infobox_items[el.id] = el;
    });
    document.querySelectorAll('#ID [data-pts-infobox]').forEach(el => {
        el.addEventListener('mouseenter', () => {
            if (!infobox_stick) {
                replace_infobox(el);
                if (infobox.scrollHeight > infobox.clientHeight || infobox_body.scrollWidth > infobox_body.clientWidth) {
                    infobox_hint.style.display = 'block';
                } else {
                    infobox_hint.style.display = 'none';
                }
            }
        });
        el.addEventListener('mouseleave', () => {
            if (!infobox_stick) {
                hide_infobox();
            }
        });
        el.addEventListener('click', (e) => {
            if (!infobox_stick) {
                infobox_stick = el.dataset.ptsInfobox;
                infobox.style.opacity = 1;
                infobox_stick = el.dataset.ptsInfobox;
                infobox_hint.style.display = 'none';
                replace_infobox(el);
                e.stopPropagation();
            } else if (infobox_stick === el.dataset.ptsInfobox) {
                hide_infobox();
                e.stopPropagation();
            } else {
                infobox_stick = el.dataset.ptsInfobox;
                replace_infobox(el);
                infobox_hint.style.display = 'none';
                e.stopPropagation();
            }
        });
    });
})();


(function () {
    // Detect vscode dark mode (not reliably detectable from css directly)
    function getLuminance(hex) {
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        const a = [r, g, b].map(function (v) {
            v /= 255;
            return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
        });
        return a[0] * 0.2126 + a[1] * 0.7152 + a[2] * 0.0722;
    }
    const vscode_background_color = getComputedStyle(document.documentElement).getPropertyValue('--vscode-editor-background');
    const container = document.querySelectorAll('#ID')[0];
    if (vscode_background_color) {
        if (getLuminance(vscode_background_color) < 0.5) {
            document.body.setAttribute('theme', 'dark');
        } else {
            document.body.setAttribute('theme', 'light');
        }
    }
})();
