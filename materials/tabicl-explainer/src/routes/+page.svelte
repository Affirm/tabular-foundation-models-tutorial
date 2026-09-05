<script lang="ts">
	import { onMount, tick } from 'svelte';
	import Topbar from '~/components/Topbar.svelte';
	import Sankey from '~/components/Sankey.svelte';
	import AttentionMatrix from '~/components/AttentionMatrix.svelte';
	import ProbabilityBars from '~/components/ProbabilityBars.svelte';
	import Textbook from '~/components/textbook/Textbook.svelte';
	import {
		CLASS_NAMES,
		CONTEXT,
		FEATURES,
		QUERIES,
		loadTabICL,
		type Inspection,
		type LoadStatus,
		type MatrixSample,
		type TabICLModel
	} from '~/lib/tabicl';

	let selectedQuery = 'A';
	let model: TabICLModel | null = null;
	let cache: unknown = null;
	let inspection: Inspection | null = null;
	let loadStatus: LoadStatus = {
		phase: 'downloading',
		progress: 0,
		message: 'Preparing model download'
	};
	let columnBlock = 0;
	let rowBlock = 0;
	let iclBlock = 0;
	let attentionHead = 0;
	let attentionExpanded = false;
	let textbookOpen = true;
	let guidePage = 0;
	const DESIGN_WIDTH = 1880;
	let fitScale = 1;
	let designHeight = 900;

	$: query = QUERIES.find((item) => item.id === selectedQuery) ?? QUERIES[0];
	$: selectedAttention = inspection?.iclBlocks[iclBlock]?.attention ?? null;
	$: selectedColumn = inspection?.columnBlocks[columnBlock]?.activation ?? null;
	$: selectedRow = inspection?.rowBlocks[rowBlock]?.tokens ?? null;
	$: revision = `${selectedQuery}-${columnBlock}-${rowBlock}-${iclBlock}-${attentionHead}-${loadStatus.phase}`;

	async function runInspection(activeModel: TabICLModel, activeCache: unknown, queryId: string) {
		const selected = QUERIES.find((item) => item.id === queryId) ?? QUERIES[0];
		inspection = null;
		loadStatus = { phase: 'running', progress: 0.98, message: `Running real Query ${queryId}` };
		await tick();
		await new Promise((resolve) => setTimeout(resolve, 20));
		try {
			inspection = activeModel.inspectQuery(activeCache, [...selected.x], 3);
			loadStatus = { phase: 'ready', progress: 1, message: 'nanoTabICL trace ready' };
		} catch (error) {
			loadStatus = {
				phase: 'error',
				progress: 0,
				message: error instanceof Error ? error.message : 'Model inference failed'
			};
		}
	}

	$: if (model && cache && selectedQuery) {
		runInspection(model, cache, selectedQuery);
	}

	onMount(async () => {
		try {
			const loaded = await loadTabICL((status) => (loadStatus = status));
			model = loaded.model;
			cache = loaded.cache;
		} catch (error) {
			loadStatus = {
				phase: 'error',
				progress: 0,
				message: error instanceof Error ? error.message : 'Could not load TabICL v2'
			};
		}
	});

	onMount(() => {
		const fitToViewport = () => {
			fitScale = Math.min(1, window.innerWidth / DESIGN_WIDTH);
			designHeight = window.innerHeight / fitScale;
		};
		fitToViewport();
		window.addEventListener('resize', fitToViewport);
		return () => window.removeEventListener('resize', fitToViewport);
	});

	function heatColor(value: number, sample: MatrixSample | null, purple = false) {
		if (!sample || !Number.isFinite(value)) return '#f3f4f6';
		const extent = Math.max(Math.abs(sample.stats.min), Math.abs(sample.stats.max), 1e-6);
		const amount = Math.min(1, Math.abs(value) / extent);
		const positive = purple ? [124, 58, 237] : [59, 130, 246];
		const negative = [239, 68, 68];
		const target = value >= 0 ? positive : negative;
		return `rgb(${target.map((channel) => Math.round(255 + (channel - 255) * amount)).join(',')})`;
	}

	function stageClass(page: number) {
		if (!textbookOpen || guidePage === 0) return '';
		return guidePage === page ? 'spotlit' : 'muted';
	}

	function cycle(value: number, delta: number, length: number) {
		return (value + delta + length) % length;
	}
</script>

<svelte:head>
	<title>TabICL Explainer: Tabular In-Context Learning, Visually Explained</title>
	<meta
		name="description"
		content="An educational nanoTabICL visualization using mapped official TabICLv2 classifier weights and Iris data."
	/>
</svelte:head>

<div class="fit-viewport">
<div
	class="app-shell"
	class:attention-expanded={attentionExpanded}
	style={`--fit-scale:${fitScale};--design-height:${designHeight}px`}
>
	<header><Topbar bind:selectedQuery {loadStatus} /></header>
	<main>
		<div class="architecture resize-watch">
			<Sankey {revision} />

			<section class={`stage raw-stage ${stageClass(1)}`}>
				<div class="stage-title">IRIS TABLE</div>
				<div class="stage-subtitle">real UCI records · cm</div>
				<div class="table-card" data-flow-node>
					<div class="feature-head">
						<span>ID</span>
						{#each FEATURES as feature}<span title={feature}>{feature.replace(' ', '\n')}</span>{/each}
						<span>y</span>
					</div>
					{#each CONTEXT as row}
						<div class="data-row">
							<b>{row.id}</b>
							{#each row.x as value}<span>{value.toFixed(1)}</span>{/each}
							<em class={`class-${row.y}`}>{CLASS_NAMES[row.y].slice(0, 3)}</em>
						</div>
					{/each}
					<div class="data-row query-row">
						<b>Q</b>
						{#each query.x as value}<span>{value.toFixed(1)}</span>{/each}
						<em>?</em>
					</div>
				</div>
				<div class="truth-note">Query {selectedQuery} is visibly unlabeled</div>
			</section>

			<section class={`stage preprocess-stage ${stageClass(2)}`}>
				<div class="stage-title">PREPROCESS + EMBED</div>
				<div class="stage-subtitle">live query path</div>
				<div class="preprocess-flow" data-flow-node>
					<div class="operation">
						<span>μ, σ</span>
						<b>Standardize</b>
						<small>context statistics</small>
					</div>
					<div class="tiny-arrow">↓</div>
					<div class="operation">
						<span>↻</span>
						<b>Group features</b>
						<small>offsets 1 · 2 · 4</small>
					</div>
					<div class="tiny-arrow">↓</div>
					<div class="operation">
						<span>W<sub>x</sub></span>
						<b>Linear embed</b>
						<small>4 × 128</small>
					</div>
					<div class="activation-label">LIVE TENSOR SAMPLE</div>
					<div class="column-strips">
						{#each inspection?.embedding.values ?? Array(4).fill(Array(24).fill(Number.NaN)) as values, index}
							<div class="strip-row">
								<span class="strip-label">f{index + 1}</span>
								<div class="heat-strip">
									{#each values as value}
										<i
											style={`background:${heatColor(value, inspection?.embedding ?? null)}`}
											title={Number.isFinite(value) ? value.toFixed(5) : ''}
										></i>
									{/each}
								</div>
							</div>
						{/each}
					</div>
				</div>
			</section>

			<section class={`stage column-stage ${stageClass(3)}`}>
				<div class="stage-title">COLUMN TRANSFORMER</div>
				<div class="stage-subtitle">induced attention · 8 heads</div>
				<div class="block-stack" data-flow-node>
					{#each Array(3) as _, index}
						<div class="back-card" style={`transform:translate(${index * 5}px,${index * 6}px)`}></div>
					{/each}
					<div class="block-card">
						<div class="pager">
							<button on:click={() => (columnBlock = cycle(columnBlock, -1, 3))}>‹</button>
							<span>Block {columnBlock + 1} of 3</span>
							<button on:click={() => (columnBlock = cycle(columnBlock, 1, 3))}>›</button>
						</div>
						<div class="inducing">128 inducing vectors</div>
						<div class="mini-block"><span>TFM 1</span><small>QASSMax</small></div>
						<div class="mini-block"><span>TFM 2</span><small>cross-attention</small></div>
						<div class="symbolic-label">SYMBOLIC STRUCTURE</div>
						<div class="column-strips live">
							{#each selectedColumn?.values ?? Array(4).fill(Array(24).fill(Number.NaN)) as values, index}
								<div class="strip-row">
									<span class="strip-label">f{index + 1}</span>
									<div class="heat-strip">
										{#each values as value}
											<i
												style={`background:${heatColor(value, selectedColumn, true)}`}
												title={Number.isFinite(value) ? value.toFixed(5) : ''}
											></i>
										{/each}
									</div>
								</div>
							{/each}
						</div>
						<div class="activation-label">LIVE BLOCK OUTPUT</div>
					</div>
				</div>
			</section>

			<section class={`stage row-stage ${stageClass(4)}`}>
				<div class="stage-title">ROW TRANSFORMER</div>
				<div class="stage-subtitle">feature mixing + CLS compression</div>
				<div class="row-card" data-flow-node>
					<div class="pager">
						<button on:click={() => (rowBlock = cycle(rowBlock, -1, 3))}>‹</button>
						<span>Block {rowBlock + 1} of 3</span>
						<button on:click={() => (rowBlock = cycle(rowBlock, 1, 3))}>›</button>
					</div>
					<div class="token-set">
						{#each Array(4) as _, index}<div class="cls">CLS {index + 1}</div>{/each}
						{#each Array(rowBlock === 2 ? 0 : 4) as _, index}<div class="feature-token">f{index + 1}</div>{/each}
					</div>
					<div class="row-attention">8-head self-attention + MLP</div>
					<div class="selected-token-strip">
						{#each selectedRow?.values?.[0] ?? Array(24).fill(Number.NaN) as value}
							<i
								style={`background:${heatColor(value, selectedRow, true)}`}
								title={Number.isFinite(value) ? value.toFixed(5) : ''}
							></i>
						{/each}
					</div>
					<div class="activation-label">LIVE FIRST TOKEN SAMPLE</div>
				</div>
			</section>

			<section class={`stage vectors-stage ${stageClass(4)}`}>
				<div class="stage-title">ROW VECTORS</div>
				<div class="stage-subtitle">4 CLS × 128 → 512</div>
				<div class="vector-column" data-flow-node>
					{#each CONTEXT as row}
						<div class="row-vector">
							<span class="vector-label">{row.id}</span><i class="symbolic-vector class-vector-{row.y}"></i>
						</div>
					{/each}
					<div class="row-vector query-vector">
						<span class="vector-label">Q</span>
						<div class="query-live-vector">
							{#each inspection?.rowVector.values?.[0] ?? Array(32).fill(Number.NaN) as value}
								<i
									style={`background:${heatColor(value, inspection?.rowVector ?? null)}`}
									title={Number.isFinite(value) ? value.toFixed(5) : ''}
								></i>
							{/each}
						</div>
					</div>
					<div class="vector-key">E = symbolic · Q = live sample</div>
				</div>
			</section>

			<section class={`stage icl-stage ${stageClass(5)}`}>
				<div class="stage-title">ICL TRANSFORMER × 12</div>
				<div class="stage-subtitle">query-to-context routing</div>
				<div class="icl-stack" data-flow-node>
					{#each Array(8) as _, index}
						<div class="head-back" style={`transform:translate(${index * 3}px,${index * 4}px)`}></div>
					{/each}
					<div class="icl-card">
						<div class="dual-pager">
							<div class="pager">
								<button on:click={() => (iclBlock = cycle(iclBlock, -1, 12))}>‹</button>
								<span>Block {iclBlock + 1} / 12</span>
								<button on:click={() => (iclBlock = cycle(iclBlock, 1, 12))}>›</button>
							</div>
							<div class="pager">
								<button on:click={() => (attentionHead = cycle(attentionHead, -1, 8))}>‹</button>
								<span>Head {attentionHead + 1} / 8</span>
								<button on:click={() => (attentionHead = cycle(attentionHead, 1, 8))}>›</button>
							</div>
						</div>
						<AttentionMatrix
							raw={selectedAttention?.rawLogits?.[attentionHead] ?? null}
							scaled={selectedAttention?.scaledLogits?.[attentionHead] ?? null}
							weights={selectedAttention?.heads?.[attentionHead] ?? null}
							bind:expanded={attentionExpanded}
						/>
						<p class="routing-note">Attention routes information from E1–E12 into Q; it is not importance.</p>
						<div class="icl-output">
							{#each inspection?.iclBlocks?.[iclBlock]?.activation.values?.[0] ?? Array(32).fill(Number.NaN) as value}
								<i
									style={`background:${heatColor(value, inspection?.iclBlocks?.[iclBlock]?.activation ?? null, true)}`}
									title={Number.isFinite(value) ? value.toFixed(5) : ''}
								></i>
							{/each}
						</div>
						<div class="activation-label">LIVE QUERY VECTOR · 32 OF 512 DIMENSIONS</div>
					</div>
				</div>
			</section>

			<section class={`stage output-stage ${stageClass(6)}`}>
				<div class="stage-title">OUTPUT PROBABILITIES</div>
				<div class="stage-subtitle">LayerNorm · MLP · softmax</div>
				<div class="output-card" data-flow-node>
					<div class="output-vector"></div>
					<span class="output-arrow">→</span>
					<ProbabilityBars probabilities={inspection?.output.probabilities ?? null} labels={CLASS_NAMES} />
					{#if inspection}
						<div class="prediction">
							Query {selectedQuery} → {CLASS_NAMES[
								inspection.output.probabilities.indexOf(Math.max(...inspection.output.probabilities))
							]}
						</div>
					{:else}
						<div class="prediction blank">Waiting for real model output</div>
					{/if}
				</div>
			</section>
		</div>
	</main>
	<footer class="provenance">
		<div>
			<strong>Model provenance</strong>
			<span>Mapped official TabICLv2 weights; SHA-256 verified before loading. The 12-row Iris trace is an out-of-regime illustration because the documented pretraining range starts at 300 rows.</span>
		</div>
	</footer>
	<Textbook bind:open={textbookOpen} bind:currentPage={guidePage} />
</div>
</div>

<style lang="scss">
	.fit-viewport {
		width: 100vw;
		height: 100vh;
		overflow: hidden;
		background: white;
	}
	.app-shell {
		width: 1880px;
		height: var(--design-height);
		min-width: 1880px;
		transform: scale(var(--fit-scale));
		transform-origin: top left;
		background: white;
		color: theme('colors.gray.700');
	}
	header {
		position: fixed;
		z-index: 300;
		top: 0;
		left: 0;
		width: 1880px;
		min-width: 1880px;
	}
	main {
		position: absolute;
		inset: 68px 0 72px;
		overflow-x: auto;
		overflow-y: hidden;
	}
	.architecture {
		position: relative;
		display: grid;
		grid-template-columns: 285px 180px 210px 205px 140px 385px 250px;
		gap: 26px;
		align-items: center;
		min-width: 1860px;
		height: 100%;
		padding: 0 32px 40px;
	}
	.stage {
		position: relative;
		z-index: 10;
		height: min(720px, calc(100vh - 130px));
		display: flex;
		flex-direction: column;
		justify-content: center;
		transition:
			opacity 0.35s,
			filter 0.35s,
			transform 0.35s;
		&.muted {
			opacity: 0.14;
			filter: grayscale(0.6);
		}
		&.spotlit {
			transform: translateY(-3px);
			.stage-title {
				color: theme('colors.blue.600');
			}
			> div[data-flow-node] {
				box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.24);
			}
		}
	}
	.stage-title {
		height: 35px;
		display: flex;
		align-items: end;
		font-size: 0.78rem;
		font-weight: 600;
		letter-spacing: 0.035em;
		color: theme('colors.gray.500');
		white-space: nowrap;
	}
	.stage-subtitle {
		height: 40px;
		padding-top: 0.25rem;
		font-size: 0.65rem;
		color: theme('colors.gray.400');
		white-space: nowrap;
	}
	.table-card,
	.preprocess-flow,
	.block-card,
	.row-card,
	.vector-column,
	.icl-card,
	.output-card {
		position: relative;
		background: white;
		border: 1px solid theme('colors.gray.100');
		border-radius: 0.55rem;
		box-shadow: 0 3px 12px rgba(17, 24, 39, 0.06);
	}
	.table-card {
		padding: 0.5rem;
		font-family: monospace;
		font-size: 0.6rem;
	}
	.feature-head,
	.data-row {
		display: grid;
		grid-template-columns: 28px repeat(4, 1fr) 30px;
		gap: 3px;
		align-items: center;
		text-align: right;
		min-height: 25px;
	}
	.feature-head {
		color: theme('colors.gray.400');
		font-family: sans-serif;
		font-size: 0.48rem;
		line-height: 0.9;
		border-bottom: 1px solid theme('colors.gray.100');
		span:first-child {
			text-align: left;
		}
	}
	.data-row {
		border-bottom: 1px solid theme('colors.gray.50');
		b {
			text-align: left;
			color: theme('colors.gray.500');
		}
		em {
			padding: 0.12rem 0.22rem;
			border-radius: 0.2rem;
			text-align: center;
			font-style: normal;
			color: theme('colors.gray.500');
			background: theme('colors.gray.100');
		}
		.class-0 {
			background: theme('colors.blue.50');
			color: theme('colors.blue.600');
		}
		.class-1 {
			background: theme('colors.purple.50');
			color: theme('colors.purple.600');
		}
		.class-2 {
			background: theme('colors.rose.50');
			color: theme('colors.rose.600');
		}
	}
	.query-row {
		margin-top: 0.35rem;
		border: 1px dashed theme('colors.purple.300');
		border-radius: 0.3rem;
		padding: 0 0.2rem;
		background: theme('colors.purple.50');
		b,
		em {
			color: theme('colors.purple.700');
		}
	}
	.truth-note,
	.vector-key {
		margin-top: 0.4rem;
		text-align: center;
		font-size: 0.58rem;
		color: theme('colors.gray.400');
	}
	.preprocess-flow {
		padding: 0.8rem;
	}
	.operation {
		display: grid;
		grid-template-columns: 30px 1fr;
		align-items: center;
		padding: 0.45rem;
		border: 1px solid theme('colors.gray.100');
		border-radius: 0.35rem;
		span {
			grid-row: 1 / 3;
			font-family: serif;
			color: theme('colors.blue.600');
		}
		b {
			font-size: 0.72rem;
			font-weight: 500;
		}
		small {
			font-size: 0.55rem;
			color: theme('colors.gray.400');
		}
	}
	.tiny-arrow {
		height: 18px;
		text-align: center;
		color: theme('colors.gray.300');
	}
	.activation-label,
	.symbolic-label {
		margin: 0.65rem 0 0.3rem;
		font-size: 0.48rem;
		font-weight: 700;
		letter-spacing: 0.12em;
		color: theme('colors.blue.500');
	}
	.symbolic-label {
		color: theme('colors.gray.400');
	}
	.column-strips {
		display: flex;
		flex-direction: column;
		gap: 4px;
	}
	.strip-row {
		display: flex;
		align-items: center;
		gap: 4px;
		.strip-label {
			width: 12px;
			font-family: monospace;
			font-size: 0.5rem;
			color: theme('colors.gray.400');
		}
	}
	.heat-strip,
	.query-live-vector,
	.selected-token-strip,
	.icl-output {
		display: flex;
		height: 9px;
		overflow: hidden;
		border-radius: 2px;
		i {
			width: 5px;
			flex: 1;
			min-width: 1px;
		}
	}
	.block-stack,
	.icl-stack {
		position: relative;
	}
	.back-card,
	.head-back {
		position: absolute;
		inset: 0;
		border: 1px solid theme('colors.gray.100');
		border-radius: 0.55rem;
		background: white;
		box-shadow: 0 2px 7px rgba(17, 24, 39, 0.05);
	}
	.block-card {
		min-height: 305px;
		padding: 0.85rem;
		z-index: 5;
	}
	.pager {
		display: flex;
		justify-content: center;
		align-items: center;
		gap: 0.45rem;
		font-size: 0.62rem;
		color: theme('colors.gray.500');
		button {
			width: 20px;
			height: 20px;
			border: 1px solid theme('colors.gray.200');
			border-radius: 0.25rem;
			color: theme('colors.gray.400');
		}
	}
	.inducing {
		margin: 1rem 0 0.45rem;
		padding: 0.45rem;
		text-align: center;
		font-size: 0.62rem;
		color: theme('colors.red.500');
		border: 1px dashed theme('colors.red.200');
	}
	.mini-block {
		display: flex;
		justify-content: space-between;
		margin-top: 0.45rem;
		padding: 0.55rem;
		border: 1px solid theme('colors.purple.100');
		background: theme('colors.purple.50');
		font-size: 0.65rem;
		color: theme('colors.purple.700');
		small {
			color: theme('colors.purple.400');
		}
	}
	.row-card {
		padding: 0.85rem;
		min-height: 260px;
	}
	.token-set {
		display: flex;
		flex-wrap: wrap;
		gap: 4px;
		margin: 1.2rem 0;
		> div {
			width: calc(25% - 3px);
			padding: 0.35rem 0;
			text-align: center;
			font-family: monospace;
			font-size: 0.53rem;
			border-radius: 0.2rem;
		}
		.cls {
			color: theme('colors.red.600');
			background: theme('colors.red.50');
			border: 1px solid theme('colors.red.100');
		}
		.feature-token {
			color: theme('colors.blue.600');
			background: theme('colors.blue.50');
		}
	}
	.row-attention {
		padding: 0.65rem 0.2rem;
		text-align: center;
		font-size: 0.6rem;
		color: theme('colors.purple.600');
		border-top: 1px solid theme('colors.gray.100');
		border-bottom: 1px solid theme('colors.gray.100');
	}
	.selected-token-strip {
		margin-top: 1rem;
	}
	.vector-column {
		padding: 0.6rem 0.7rem;
	}
	.row-vector {
		height: 27px;
		display: flex;
		align-items: center;
		gap: 5px;
		.vector-label {
			width: 24px;
			text-align: right;
			font-family: monospace;
			font-size: 0.58rem;
			color: theme('colors.gray.500');
		}
		.symbolic-vector {
			width: 70px;
			height: 9px;
			border: 1px dashed theme('colors.gray.300');
			background: linear-gradient(90deg, theme('colors.gray.50'), theme('colors.gray.100'));
		}
		.class-vector-0 {
			border-color: theme('colors.blue.200');
		}
		.class-vector-1 {
			border-color: theme('colors.purple.200');
		}
		.class-vector-2 {
			border-color: theme('colors.rose.200');
		}
	}
	.query-vector {
		margin-top: 0.25rem;
		padding-top: 0.3rem;
		border-top: 1px solid theme('colors.gray.100');
		.vector-label {
			color: theme('colors.purple.700');
			font-weight: 700;
		}
	}
	.query-live-vector {
		width: 70px;
	}
	.icl-stack {
		z-index: 20;
	}
	.head-back {
		z-index: 1;
	}
	.icl-card {
		z-index: 15;
		min-height: 330px;
		padding: 0.85rem;
	}
	.dual-pager {
		display: flex;
		justify-content: space-between;
		margin-bottom: 1.1rem;
	}
	.routing-note {
		margin: 1rem auto 0;
		max-width: 280px;
		text-align: center;
		font-size: 0.58rem;
		line-height: 1.35;
		color: theme('colors.gray.400');
	}
	.icl-output {
		margin-top: 1.15rem;
	}
	.output-card {
		min-height: 260px;
		padding: 1.2rem;
		display: flex;
		flex-direction: column;
		justify-content: center;
	}
	.output-vector {
		width: 12px;
		height: 75px;
		margin: 0 auto;
		border: 1px solid theme('colors.blue.200');
		background: linear-gradient(theme('colors.blue.50'), theme('colors.purple.100'));
	}
	.output-arrow {
		text-align: center;
		padding: 0.45rem 0;
		color: theme('colors.gray.300');
		transform: rotate(90deg);
	}
	.prediction {
		margin-top: 1.2rem;
		padding-top: 0.8rem;
		border-top: 1px solid theme('colors.gray.100');
		font-size: 0.72rem;
		text-align: center;
		color: theme('colors.purple.700');
		font-weight: 600;
		&.blank {
			color: theme('colors.gray.400');
			font-weight: 400;
		}
	}
	.attention-expanded {
		.stage:not(.icl-stage) {
			opacity: 0.15;
		}
	}
	footer.provenance {
		position: fixed;
		left: 0;
		right: 0;
		bottom: 0;
		z-index: 200;
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 4rem;
		min-height: 64px;
		padding: 0.7rem 2rem;
		border-top: 2px solid theme('colors.blue.500');
		background: theme('colors.blue.50');
		color: theme('colors.gray.700');
		font-size: 0.82rem;
		box-shadow: 0 -8px 24px rgb(15 23 42 / 0.08);
		div {
			display: flex;
			align-items: baseline;
			gap: 0.65rem;
		}
		strong {
			color: theme('colors.blue.800');
			font-size: 0.72rem;
			letter-spacing: 0.08em;
			text-transform: uppercase;
		}
	}
</style>
