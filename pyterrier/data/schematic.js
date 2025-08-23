(function () {
    var infobox_stick = null;
    var infobox_source_el = null;
    const infobox_items = {};
    const infobox = document.querySelectorAll('#ID .infobox')[0];
    const infobox_title = document.querySelectorAll('#ID .infobox-title')[0];
    const infobox_body = document.querySelectorAll('#ID .infobox-body')[0];
    const container = document.querySelectorAll('#ID')[0];
    const is_repr_html = container.classList.contains('repr_html');
    function replace_infobox(el) {
        if (infobox_source_el !== null) {
            infobox_source_el.classList.remove('infobox-source');
            infobox_source_el = null;
        }
        infobox_body.innerHTML = '';
        infobox_title.textContent = infobox_items[el.dataset.infobox].dataset.title || '';
        infobox.style.display = 'block';
        infobox_body.appendChild(infobox_items[el.dataset.infobox]);
        if (infobox_body.querySelectorAll('.infobox-error').length > 0) {
            infobox.classList.add('infobox-outer-error');
        } else {
            infobox.classList.remove('infobox-outer-error');
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
        el.classList.add('infobox-source');
    }
    function hide_infobox() {
        if (infobox_source_el !== null) {
            infobox_source_el.classList.remove('infobox-source');
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
    document.querySelectorAll('#ID .infobox-item').forEach(el => {
        el.remove();
        el.style.display = 'block';
        infobox_items[el.id] = el;
    });
    document.querySelectorAll('#ID [data-infobox]').forEach(el => {
        el.addEventListener('mouseenter', () => {
            if (!infobox_stick) {
                replace_infobox(el);
            }
        });
        el.addEventListener('mouseleave', () => {
            if (!infobox_stick) {
                hide_infobox();
            }
        });
        el.addEventListener('click', (e) => {
            if (!infobox_stick) {
                infobox_stick = el.dataset.infobox;
                infobox.style.opacity = 1;
                infobox_stick = el.dataset.infobox;
                replace_infobox(el);
                e.stopPropagation();
            } else if (infobox_stick === el.dataset.infobox) {
                hide_infobox();
                e.stopPropagation();
            } else {
                infobox_stick = el.dataset.infobox;
                replace_infobox(el);
                e.stopPropagation();
            }
        });
    });
})();
