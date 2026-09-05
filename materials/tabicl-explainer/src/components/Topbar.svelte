<script lang="ts">
	import type { LoadStatus } from '~/lib/tabicl';

	export let selectedQuery = 'A';
	export let loadStatus: LoadStatus;
</script>

<div class="top-bar">
	<div class="logo">T<span>AB</span>ICL E<span>XPLAINER</span></div>
	<div class="interface-credit">
		<span>Interface adapted from</span>
		<a href="https://poloclub.github.io/transformer-explainer/" target="_blank" rel="noreferrer"
			>Transformer Explainer</a
		>
		<span>by Aeree Cho et al. · Polo Club of Data Science · MIT</span>
	</div>
	<div class="controls">
		<span class="control-label">IRIS QUERY</span>
		<div class="query-tabs" aria-label="Select an unlabeled Iris query">
			{#each ['A', 'B', 'C'] as query}
				<button
					class:active={selectedQuery === query}
					on:click={() => (selectedQuery = query)}
					aria-pressed={selectedQuery === query}>Query {query}</button
				>
			{/each}
		</div>
		<div class="load-state" class:ready={loadStatus.phase === 'ready'}>
			<span class="status-dot"></span>
			<span>{loadStatus.message}</span>
			{#if loadStatus.phase !== 'ready' && loadStatus.phase !== 'error'}
				<span class="percent">{Math.round(loadStatus.progress * 100)}%</span>
			{/if}
			<div class="progress"><i style={`width:${loadStatus.progress * 100}%`}></i></div>
		</div>
	</div>
	<a
		class="upstream-link"
		href="https://github.com/poloclub/transformer-explainer"
		target="_blank"
		rel="noreferrer"
		aria-label="Open the original Transformer Explainer source code"
	>
		Original interface code ↗
	</a>
</div>

<style lang="scss">
	.top-bar {
		height: 68px;
		display: flex;
		align-items: center;
		gap: 2rem;
		padding: 0.55rem 2.3rem 0.8rem;
		background: linear-gradient(to bottom, #fff 72%, rgba(255, 255, 255, 0));
	}
	.logo {
		flex-shrink: 0;
		white-space: nowrap;
		font-family: 'Jersey 10', sans-serif;
		font-size: 2rem;
		background: linear-gradient(to right, theme('colors.blue.500'), theme('colors.purple.500'));
		background-clip: text;
		-webkit-background-clip: text;
		-webkit-text-fill-color: transparent;
		span {
			font-size: 1.78rem;
		}
	}
	.interface-credit {
		display: flex;
		flex-direction: column;
		flex-shrink: 0;
		padding-left: 1rem;
		border-left: 1px solid theme('colors.gray.200');
		color: theme('colors.gray.500');
		font-size: 0.62rem;
		line-height: 1.25;
		a {
			color: theme('colors.blue.700');
			font-size: 0.72rem;
			font-weight: 700;
			text-decoration: underline;
			text-underline-offset: 2px;
		}
	}
	.controls {
		display: flex;
		align-items: center;
		gap: 0.7rem;
		flex: 1;
		min-width: 0;
	}
	.control-label {
		font-size: 0.65rem;
		letter-spacing: 0.12em;
		font-weight: 700;
		color: theme('colors.gray.400');
	}
	.query-tabs {
		display: flex;
		border: 1px solid theme('colors.gray.200');
		border-radius: 0.4rem;
		overflow: hidden;
		background: white;
		button {
			padding: 0.32rem 0.72rem;
			font-size: 0.78rem;
			color: theme('colors.gray.500');
			border-right: 1px solid theme('colors.gray.200');
			transition: 0.15s;
			&:last-child {
				border-right: 0;
			}
			&.active {
				color: theme('colors.purple.700');
				background: theme('colors.purple.50');
				font-weight: 600;
			}
		}
	}
	.load-state {
		position: relative;
		display: flex;
		align-items: center;
		gap: 0.35rem;
		min-width: 260px;
		padding: 0.35rem 0.6rem 0.42rem;
		border: 1px solid theme('colors.gray.200');
		border-radius: 0.4rem;
		color: theme('colors.gray.500');
		font-size: 0.72rem;
		.status-dot {
			width: 7px;
			height: 7px;
			border-radius: 999px;
			background: theme('colors.amber.400');
		}
		&.ready .status-dot {
			background: theme('colors.emerald.500');
		}
		.percent {
			margin-left: auto;
			font-family: monospace;
		}
		.progress {
			position: absolute;
			height: 2px;
			left: 0;
			right: 0;
			bottom: 0;
			overflow: hidden;
			i {
				display: block;
				height: 100%;
				background: linear-gradient(90deg, theme('colors.blue.400'), theme('colors.purple.500'));
				transition: width 0.15s;
			}
		}
	}
	.upstream-link {
		flex-shrink: 0;
		color: theme('colors.blue.700');
		font-size: 0.72rem;
		font-weight: 700;
		text-decoration: underline;
		text-underline-offset: 2px;
		&:hover {
			color: theme('colors.blue.900');
		}
	}
</style>
