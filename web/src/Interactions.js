import {invert4, rotate4, translate4} from "./utils.js";

/**
 * Add events listeners to window and canvas to catch mouse & key event and update accordingly the view matrix
 */
export class Interactions {
    viewMatrix = null
    canvas = null
    carousel = false

    constructor(viewMatrix, canvas) {
        this.viewMatrix = viewMatrix
        this.canvas = canvas

        this.init()
    }

    init() {
        window.addEventListener("wheel", (e) => {
            this.carousel = false;
            e.preventDefault();
            const lineHeight = 10;
            const scale = e.deltaMode == 1 ? lineHeight : e.deltaMode == 2 ? innerHeight : 1;
            let inv = invert4(this.viewMatrix);
            if (e.shiftKey) {
                inv = translate4(inv, (e.deltaX * scale) / innerWidth, (e.deltaY * scale) / innerHeight, 0,);
            } else if (e.ctrlKey || e.metaKey) {
                // inv = rotate4(inv,  (e.deltaX * scale) / innerWidth,  0, 0, 1);
                // inv = translate4(inv,  0, (e.deltaY * scale) / innerHeight, 0);
                // let preY = inv[13];
                inv = translate4(inv, 0, 0, (-10 * (e.deltaY * scale)) / innerHeight,);
                // inv[13] = preY;
            } else {
                let d = 4;
                inv = translate4(inv, 0, 0, d);
                inv = rotate4(inv, -(e.deltaX * scale) / innerWidth, 0, 1, 0);
                inv = rotate4(inv, (e.deltaY * scale) / innerHeight, 1, 0, 0);
                inv = translate4(inv, 0, 0, -d);
            }
            this.viewMatrix = invert4(inv);
        }, {passive: false},);

        let startX, startY, down;
        this.canvas.addEventListener("mousedown", (e) => {
            this.carousel = false;
            e.preventDefault();
            startX = e.clientX;
            startY = e.clientY;
            down = e.ctrlKey || e.metaKey ? 2 : 1;
        });
        this.canvas.addEventListener("contextmenu", (e) => {
            this.carousel = false;
            e.preventDefault();
            startX = e.clientX;
            startY = e.clientY;
            down = 2;
        });

        this.canvas.addEventListener("mousemove", (e) => {
            e.preventDefault();
            if (down == 1) {
                let inv = invert4(this.viewMatrix);
                let dx = (5 * (e.clientX - startX)) / innerWidth;
                let dy = (5 * (e.clientY - startY)) / innerHeight;
                let d = 4;

                inv = translate4(inv, 0, 0, d);
                inv = rotate4(inv, dx, 0, 1, 0);
                inv = rotate4(inv, -dy, 1, 0, 0);
                inv = translate4(inv, 0, 0, -d);
                // let postAngle = Math.atan2(inv[0], inv[10])
                // inv = rotate4(inv, postAngle - preAngle, 0, 0, 1)
                // console.log(postAngle)
                this.viewMatrix = invert4(inv);

                startX = e.clientX;
                startY = e.clientY;
            } else if (down == 2) {
                let inv = invert4(this.viewMatrix);
                // inv = rotateY(inv, );
                // let preY = inv[13];
                inv = translate4(inv, (-10 * (e.clientX - startX)) / innerWidth, 0, (10 * (e.clientY - startY)) / innerHeight,);
                // inv[13] = preY;
                this.viewMatrix = invert4(inv);

                startX = e.clientX;
                startY = e.clientY;
            }
        });
        this.canvas.addEventListener("mouseup", (e) => {
            e.preventDefault();
            down = false;
            startX = 0;
            startY = 0;
        });

        let altX = 0, altY = 0;
        this.canvas.addEventListener("touchstart", (e) => {
            e.preventDefault();
            if (e.touches.length === 1) {
                this.carousel = false;
                startX = e.touches[0].clientX;
                startY = e.touches[0].clientY;
                down = 1;
            } else if (e.touches.length === 2) {
                // console.log('beep')
                this.carousel = false;
                startX = e.touches[0].clientX;
                altX = e.touches[1].clientX;
                startY = e.touches[0].clientY;
                altY = e.touches[1].clientY;
                down = 1;
            }
        }, {passive: false},);
        this.canvas.addEventListener("touchmove", (e) => {
            e.preventDefault();
            if (e.touches.length === 1 && down) {
                let inv = invert4(this.viewMatrix);
                let dx = (4 * (e.touches[0].clientX - startX)) / innerWidth;
                let dy = (4 * (e.touches[0].clientY - startY)) / innerHeight;

                let d = 4;
                inv = translate4(inv, 0, 0, d);
                // inv = translate4(inv,  -x, -y, -z);
                // inv = translate4(inv,  x, y, z);
                inv = rotate4(inv, dx, 0, 1, 0);
                inv = rotate4(inv, -dy, 1, 0, 0);
                inv = translate4(inv, 0, 0, -d);

                this.viewMatrix = invert4(inv);

                startX = e.touches[0].clientX;
                startY = e.touches[0].clientY;
            } else if (e.touches.length === 2) {
                // alert('beep')
                const dtheta = Math.atan2(startY - altY, startX - altX) - Math.atan2(e.touches[0].clientY - e.touches[1].clientY, e.touches[0].clientX - e.touches[1].clientX,);
                const dscale = Math.hypot(startX - altX, startY - altY) / Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY,);
                const dx = (e.touches[0].clientX + e.touches[1].clientX - (startX + altX)) / 2;
                const dy = (e.touches[0].clientY + e.touches[1].clientY - (startY + altY)) / 2;
                let inv = invert4(this.viewMatrix);
                // inv = translate4(inv,  0, 0, d);
                inv = rotate4(inv, dtheta, 0, 0, 1);

                inv = translate4(inv, -dx / innerWidth, -dy / innerHeight, 0);

                // let preY = inv[13];
                inv = translate4(inv, 0, 0, 3 * (1 - dscale));
                // inv[13] = preY;

                this.viewMatrix = invert4(inv);

                startX = e.touches[0].clientX;
                altX = e.touches[1].clientX;
                startY = e.touches[0].clientY;
                altY = e.touches[1].clientY;
            }
        }, {passive: false},);
        this.canvas.addEventListener("touchend", (e) => {
            e.preventDefault();
            down = false;
            startX = 0;
            startY = 0;
        }, {passive: false},);

    }
}