<script lang="ts">
	import AttentionRow from '~/components/common/AttentionRow.svelte';
	import * as d3 from 'd3';

	export let raw: number[] | null = null;
	export let scaled: number[] | null = null;
	export let weights: number[] | null = null;
	export let expanded = false;

	const empty: number[] = Array(12).fill(Number.NaN);
	$: rawValues = raw ?? empty;
	$: scaledValues = scaled ?? empty;
	$: weightValues = weights ?? empty;

	const createSignedColor = (values: number[]) => {
		const extentValues = d3.extent(values.filter(Number.isFinite));
		const extent = Math.max(
			Math.abs(extentValues[0] ?? 1),
			Math.abs(extentValues[1] ?? 1),
			1e-6
		);
		const scale = d3
			.scaleLinear<string>()
			.domain([-extent, 0, extent])
			.range(['#ef4444', '#fff', '#7e22ce']);

		return (value: number) => (Number.isFinite(value) ? scale(value) : '#f3f4f6');
	};

	$: signedColor = createSignedColor(raw ?? []);
	$: scaledColor = createSignedColor(scaled ?? []);
	const weightColor = (value: number) =>
		Number.isFinite(value) ? d3.interpolateRgb('#ffffff', '#6d28d9')(Math.min(1, value * 4)) : '#f3f4f6';
</script>

<button
	type="button"
	class="attention-calculation"
	class:expanded
	on:click={() => (expanded = !expanded)}
	aria-expanded={expanded}
>
	{#if expanded}
		<div class="calculation-step">
			<AttentionRow values={rawValues} colorScale={signedColor} />
			<div class="matrix-label">Pre-QASS dot product</div>
			<code>Q · Kᵀ / √d</code>
		</div>
		<span class="arrow">→</span>
		<div class="calculation-step">
			<AttentionRow values={scaledValues} colorScale={scaledColor} />
			<div class="matrix-label">QASSMax-scaled logits</div>
			<code>QASSMax(Q) · Kᵀ / √d</code>
		</div>
		<span class="arrow">→</span>
		<div class="calculation-step">
			<AttentionRow values={weightValues} colorScale={weightColor} />
			<div class="matrix-label">Softmax weights</div>
			<code>softmax(scaled logits)</code>
		</div>
	{:else}
		<div class="calculation-step compact">
			<div class="row-labels">
				{#each Array(12) as _, index}<span>E{index + 1}</span>{/each}
			</div>
			<AttentionRow values={weightValues} colorScale={weightColor} cellSize={17} />
			<div class="matrix-label">Query Q attention routing <span>↗ expand</span></div>
		</div>
	{/if}
</button>

<style lang="scss">
	.attention-calculation {
		position: relative;
		display: flex;
		align-items: center;
		gap: 0.7rem;
		padding: 0.85rem;
		border-radius: 0.55rem;
		background: white;
		transition: 0.25s;
		cursor: pointer;
		box-shadow: 0 3px 12px rgba(17, 24, 39, 0.08);
		border: 1px solid theme('colors.gray.100');
		&:hover {
			background: theme('colors.gray.50');
		}
		&.expanded {
			transform: translateX(-230px);
			z-index: 80;
			box-shadow: 0 14px 34px rgba(17, 24, 39, 0.16);
		}
	}
	.calculation-step {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 0.35rem;
		&.compact {
			min-width: 255px;
		}
	}
	.row-labels {
		width: 237px;
		display: grid;
		grid-template-columns: repeat(12, 1fr);
		font-size: 0.48rem;
		font-family: monospace;
		color: theme('colors.gray.400');
	}
	.matrix-label {
		font-size: 0.72rem;
		color: theme('colors.gray.500');
		white-space: nowrap;
		span {
			color: theme('colors.purple.500');
		}
	}
	code {
		font-size: 0.58rem;
		color: theme('colors.gray.400');
		white-space: nowrap;
	}
	.arrow {
		color: theme('colors.gray.300');
	}
</style>
