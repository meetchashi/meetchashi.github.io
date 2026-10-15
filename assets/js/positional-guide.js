(function () {
    'use strict';
    document.querySelectorAll('[data-rope-demo]').forEach(function (demo) {
        const gap = demo.querySelector('[data-gap]');
        const shift = demo.querySelector('[data-shift]');
        function needle(selector, cx, position) {
            const angle = position * Math.PI / 6;
            const line = demo.querySelector(selector);
            line.setAttribute('x2', (cx + 95 * Math.cos(angle)).toFixed(3));
            line.setAttribute('y2', (140 - 95 * Math.sin(angle)).toFixed(3));
        }
        function update() {
            const g = Number(gap.value), s = Number(shift.value), k = 1, q = k + g;
            needle('[data-q-one]', 175, q);
            needle('[data-k-one]', 175, k);
            needle('[data-q-two]', 545, q + s);
            needle('[data-k-two]', 545, k + s);
            demo.querySelector('[data-original-label]').textContent = 'Query at ' + q + ' · key at ' + k;
            demo.querySelector('[data-shift-label]').textContent = 'Query at ' + (q + s) + ' · key at ' + (k + s);
            demo.querySelector('[data-rope-result]').textContent = g + ' step' + (g === 1 ? '' : 's') + ' apart · the arrows are ' + (g * 30) + '° apart · alignment score ' + Math.cos(g * Math.PI / 6).toFixed(3) + ' in both pictures.';
            demo.querySelector('desc').textContent = 'Original q at ' + q + ', k at ' + k + '; shifted q at ' + (q + s) + ', k at ' + (k + s) + '. ' + demo.querySelector('[data-rope-result]').textContent;
        }
        gap.addEventListener('input', update);
        shift.addEventListener('input', update);
        update();
        demo.querySelector('[data-gap]').closest('.pe-controls').hidden = false;
    });
}());
