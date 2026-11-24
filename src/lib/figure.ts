const prefersReducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

export abstract class FigureElement extends HTMLElement {
	private cleanups: (() => void)[] = [];
	private frame = 0;
	private animation = 0;
	private visibility?: IntersectionObserver;

	protected abstract setup(): void;

	connectedCallback() {
		this.setup();
	}

	disconnectedCallback() {
		this.stop();
		for (const fn of this.cleanups.splice(0)) fn();
	}

	protected onCleanup(fn: () => void) {
		this.cleanups.push(fn);
	}

	protected $<T extends Element = HTMLElement>(selector: string, root: ParentNode = this): T {
		const el = root.querySelector<T>(selector);
		if (!el) throw new Error(`${this.localName}: missing ${selector}`);
		return el;
	}

	protected num(key: string): number {
		return Number(this.dataset[key]);
	}

	/** Set the text of a `<Readout>` value. */
	protected out(key: string, text: string) {
		this.$(`[data-out="${key}"]`).textContent = text;
	}

	protected on<K extends keyof HTMLElementEventMap>(
		target: HTMLElement,
		type: K,
		listener: (event: HTMLElementEventMap[K]) => void,
	) {
		const handle = (event: HTMLElementEventMap[K]) => {
			// A reader's input takes over from pending autoplay.
			this.visibility?.disconnect();
			listener(event);
		};
		target.addEventListener(type, handle);
		this.onCleanup(() => target.removeEventListener(type, handle));
	}

	/** Call `fn` with the chosen value whenever a `<Seg name>` changes. */
	protected onSeg(name: string, fn: (value: string) => void) {
		const seg = this.$(`.seg[data-name="${name}"]`);
		this.on(seg, "click", (e) => {
			const button = (e.target as Element).closest("button");
			if (!button?.dataset.value) return;
			for (const b of seg.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b === button));
			fn(button.dataset.value);
		});
	}

	/** Run `fn` once, the first time half of the figure — or half a screen, for a
	 *  figure taller than the viewport — is on screen. */
	protected whenVisible(fn: () => void) {
		if (prefersReducedMotion()) return fn();
		const io = new IntersectionObserver(
			([entry]) => {
				const need = Math.min(entry.boundingClientRect.height, entry.rootBounds?.height ?? Infinity) / 2;
				if (!entry.isIntersecting || entry.intersectionRect.height < need) return;
				io.disconnect();
				fn();
			},
			{ threshold: [0, 0.1, 0.25, 0.5, 0.75, 1] },
		);
		this.visibility = io;
		io.observe(this);
		this.onCleanup(() => io.disconnect());
	}

	/** Call `step` with progress 0..1 over `ms`; jumps to 1 under reduced motion. */
	protected tween(ms: number, step: (progress: number) => void) {
		this.stop();
		if (prefersReducedMotion() || ms <= 0) return step(1);
		const animation = this.animation;
		const start = performance.now();
		const tick = (now: number) => {
			if (animation !== this.animation) return;
			const p = Math.min(1, (now - start) / ms);
			step(p);
			if (p < 1 && animation === this.animation) this.frame = requestAnimationFrame(tick);
		};
		tick(start);
	}

	protected stop() {
		cancelAnimationFrame(this.frame);
		this.animation++;
	}
}

export function define(tag: string, element: CustomElementConstructor) {
	if (!customElements.get(tag)) customElements.define(tag, element);
}
