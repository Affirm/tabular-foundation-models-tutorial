<script lang="ts">
	export let open = true;
	export let currentPage = 0;

	const pages = [
		{
			title: 'A table becomes a prediction',
			body: 'This inspector runs a fixed single core pass with quantized official TabICLv2 weights on twelve labeled Iris examples and one unlabeled query. The context is below the documented 300 to 48K-row pretraining range. Choose Query A, B, or C; its label is never supplied. This illustrates computation, not model quality.'
		},
		{
			title: '1 · Read the table',
			body: 'Each E row is a labeled context example. Q is the selected unlabeled query. Row IDs E1–E12 and Q stay fixed through the full visualization.'
		},
		{
			title: '2 · Standardize, group, embed',
			body: 'This core pass uses context-only mean and standard deviation, feature groups at checkpoint offsets 1, 2, and 4, and a projection to 128 dimensions per group. Context tokens also receive label embeddings; the query does not. Activation colors are scaled separately for each tensor and cannot be compared across panels.'
		},
		{
			title: '3 · Column attention',
			body: 'Context rows first build three blocks of inducing summaries. The query reads cached summaries through the second attention operation of each block; it does not rebuild them. Colored strips show sampled query activations. The inducing cards are symbolic.'
		},
		{
			title: '4 · Compress each row',
			body: 'Three Row Transformer blocks combine grouped-feature tokens with four learned CLS tokens. The final CLS outputs are normalized and concatenated into a 512-dimensional row vector. Context vectors receive a second label embedding before dataset-level ICL; the query does not.'
		},
		{
			title: '5 · Route through context',
			body: 'Twelve ICL Transformer blocks let Q attend to cached context keys and values from E1–E12. Attention weights describe routing, not causal attribution. Change block or head, or expand to compare logits on a shared color scale and inspect softmax weights on their separate scale.'
		},
		{
			title: '6 · Predict the class',
			body: 'Output normalization and an MLP produce three class logits, converted to probabilities with temperature 1. This fixed core pass keeps the original feature and class order. It does not select one of the playground classifier’s eight views or use its temperature 0.9.'
		}
	];

	function previous() {
		currentPage = (currentPage - 1 + pages.length) % pages.length;
	}
	function next() {
		currentPage = (currentPage + 1) % pages.length;
	}
</script>

{#if open}
	<aside class="text-card" role="dialog" aria-label="TabICL explainer lesson">
		<div class="card-header">
			<div class="book">▤</div>
			<h3>{pages[currentPage].title}</h3>
			<button class="close" on:click={() => (open = false)} aria-label="Close lesson">×</button>
		</div>
		<div class="card-body">
			<p>{pages[currentPage].body}</p>
			{#if currentPage === 0}
				<div class="legend">
					<span><i class="symbolic"></i> symbolic structure</span>
					<span><i class="live"></i> live tensor values</span>
				</div>
			{/if}
		</div>
		<div class="navigation">
			<button on:click={previous} aria-label="Previous lesson">‹</button>
			<div class="dots">
				{#each pages as _, index}
					<button
						class:active={currentPage === index}
						on:click={() => (currentPage = index)}
						aria-label={`Lesson ${index + 1}`}
					></button>
				{/each}
			</div>
			<button on:click={next} aria-label="Next lesson">›</button>
		</div>
	</aside>
{:else}
	<button class="floating-book" on:click={() => (open = true)} aria-label="Open lesson">▤</button>
{/if}

<style lang="scss">
	.text-card {
		position: fixed;
		right: 2rem;
		bottom: 2rem;
		z-index: 1000;
		width: 510px;
		min-height: 245px;
		border-radius: 0.65rem;
		background: rgba(255, 255, 255, 0.82);
		backdrop-filter: blur(12px);
		box-shadow:
			0 12px 32px rgba(0, 0, 0, 0.12),
			0 4px 8px rgba(0, 0, 0, 0.06);
		display: flex;
		flex-direction: column;
	}
	.card-header {
		display: flex;
		align-items: center;
		gap: 0.55rem;
		padding: 0.75rem 0.9rem 0.35rem;
		h3 {
			flex: 1;
			font-size: 1rem;
			font-weight: 600;
			color: theme('colors.gray.800');
		}
		.book {
			color: theme('colors.purple.600');
			font-size: 1.25rem;
		}
		.close {
			width: 1.7rem;
			height: 1.7rem;
			border: 1px solid theme('colors.gray.200');
			border-radius: 0.35rem;
			color: theme('colors.gray.500');
		}
	}
	.card-body {
		flex: 1;
		padding: 0.5rem 1rem;
		color: theme('colors.gray.600');
		font-size: 0.92rem;
		line-height: 1.55;
	}
	.legend {
		display: flex;
		gap: 1.25rem;
		margin-top: 0.8rem;
		font-size: 0.72rem;
		color: theme('colors.gray.500');
		span {
			display: flex;
			align-items: center;
			gap: 0.35rem;
		}
		i {
			width: 1.3rem;
			height: 0.35rem;
			display: block;
			border-radius: 999px;
			&.symbolic {
				border: 1px dashed theme('colors.gray.400');
			}
			&.live {
				background: linear-gradient(90deg, theme('colors.blue.400'), theme('colors.purple.500'));
			}
		}
	}
	.navigation {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 0.4rem 0.9rem 0.75rem;
		color: theme('colors.gray.400');
		> button {
			font-size: 1.5rem;
		}
	}
	.dots {
		display: flex;
		gap: 0.35rem;
		button {
			width: 6px;
			height: 6px;
			border-radius: 999px;
			background: theme('colors.gray.300');
			&.active {
				width: 18px;
				background: theme('colors.purple.500');
			}
		}
	}
	.floating-book {
		position: fixed;
		right: 2rem;
		bottom: 2rem;
		z-index: 1000;
		width: 3.5rem;
		height: 3.5rem;
		border-radius: 999px;
		color: white;
		font-size: 1.5rem;
		background: linear-gradient(135deg, theme('colors.purple.500'), theme('colors.blue.500'));
		box-shadow: 0 5px 18px rgba(76, 29, 149, 0.3);
	}
</style>
