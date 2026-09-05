<script lang="ts">
	export let probabilities: number[] | null = null;
	export let labels: string[] = [];
	$: prediction = probabilities ? probabilities.indexOf(Math.max(...probabilities)) : -1;
</script>

<div class="probability-col">
	{#each labels as label, index}
		<div class="probability" class:predicted={prediction === index}>
			<div class="label-row">
				<span>{label}</span>
				<strong>{probabilities ? `${(probabilities[index] * 100).toFixed(2)}%` : ''}</strong>
			</div>
			<div class="track">
				<i style={`width:${probabilities ? probabilities[index] * 100 : 0}%`}></i>
			</div>
		</div>
	{/each}
</div>

<style lang="scss">
	.probability-col {
		width: 190px;
		display: flex;
		flex-direction: column;
		gap: 1rem;
	}
	.probability {
		.label-row {
			display: flex;
			justify-content: space-between;
			align-items: baseline;
			margin-bottom: 0.35rem;
			color: theme('colors.gray.500');
			font-size: 0.78rem;
			strong {
				min-width: 3.7rem;
				text-align: right;
				font-family: monospace;
				font-size: 0.72rem;
				font-weight: 500;
			}
		}
		.track {
			height: 5px;
			border-radius: 999px;
			background: theme('colors.gray.200');
			overflow: hidden;
			i {
				display: block;
				height: 100%;
				background: theme('colors.gray.400');
				transition: width 0.65s cubic-bezier(0.16, 1, 0.3, 1);
			}
		}
		&.predicted {
			.label-row {
				color: theme('colors.purple.700');
				font-weight: 600;
			}
			.track i {
				background: linear-gradient(90deg, theme('colors.blue.500'), theme('colors.purple.600'));
			}
		}
	}
</style>
