<script lang="ts">
	export let values: number[];
	export let colorScale: (value: number) => string;
	export let cellSize = 16;
	export let gap = 3;

	const displayValue = (value: number) =>
		Number.isFinite(value) ? value.toFixed(5) : String(value);
</script>

<div class="attention-row" style:gap={`${gap}px`}>
	{#each values as value}
		<span
			class="cell"
			class:missing={!Number.isFinite(value)}
			style:width={`${cellSize}px`}
			style:height={`${cellSize}px`}
			style:background-color={colorScale(value)}
			title={displayValue(value)}
			aria-hidden="true"
		></span>
	{/each}
</div>

<style>
	.attention-row {
		display: flex;
	}

	.cell {
		flex: none;
		box-sizing: border-box;
		border: 1px solid #e5e7eb;
		border-radius: 50%;
	}

	.cell.missing {
		border-color: transparent;
	}
</style>
