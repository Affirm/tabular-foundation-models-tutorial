<script lang="ts">
	import { onMount, tick } from 'svelte';
	import * as d3 from 'd3';

	export let revision = '';
	let svgEl: SVGSVGElement;
	let observer: ResizeObserver;

	const draw = async () => {
		await tick();
		if (!svgEl) return;
		const host = svgEl.parentElement?.getBoundingClientRect();
		const nodes = Array.from(document.querySelectorAll<HTMLElement>('[data-flow-node]'));
		if (!host || nodes.length < 2) return;
		const links = nodes.slice(0, -1).map((node, index) => {
			const source = node.getBoundingClientRect();
			const target = nodes[index + 1].getBoundingClientRect();
			const x1 = source.right - host.left;
			const x2 = target.left - host.left;
			const y1 = source.top + source.height / 2 - host.top;
			const y2 = target.top + target.height / 2 - host.top;
			const curve = Math.max(20, (x2 - x1) * 0.45);
			return {
				path: `M${x1},${y1} C${x1 + curve},${y1} ${x2 - curve},${y2} ${x2},${y2}`,
				index
			};
		});
		d3.select(svgEl)
			.select('g.links')
			.selectAll('path')
			.data(links)
			.join('path')
			.attr('d', (d) => d.path)
			.attr('stroke', (d) => `url(#flow-${d.index % 3})`);
	};

	onMount(() => {
		observer = new ResizeObserver(draw);
		const host = svgEl.parentElement;
		if (host) observer.observe(host);
		window.addEventListener('resize', draw);
		draw();
		return () => {
			observer.disconnect();
			window.removeEventListener('resize', draw);
		};
	});

	$: if (revision) draw();
</script>

<svg bind:this={svgEl} class="sankey-top" aria-hidden="true">
	<defs>
		<linearGradient id="flow-0"><stop stop-color="#93c5fd" /><stop offset="1" stop-color="#c4b5fd" /></linearGradient>
		<linearGradient id="flow-1"><stop stop-color="#c4b5fd" /><stop offset="1" stop-color="#a5b4fc" /></linearGradient>
		<linearGradient id="flow-2"><stop stop-color="#a5b4fc" /><stop offset="1" stop-color="#ddd6fe" /></linearGradient>
	</defs>
	<g class="links"></g>
</svg>

<style>
	.sankey-top {
		position: absolute;
		inset: 0;
		width: 100%;
		height: 100%;
		overflow: visible;
		pointer-events: none;
		z-index: 2;
	}
	:global(.sankey-top path) {
		fill: none;
		stroke-width: 7px;
		stroke-linecap: round;
		opacity: 0.28;
	}
</style>
