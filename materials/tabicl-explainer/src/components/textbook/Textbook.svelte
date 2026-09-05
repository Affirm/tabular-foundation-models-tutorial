<script lang="ts">
	export let open = true;
	export let currentPage = 0;

	const pages = [
		{
			title: 'A table becomes a prediction',
			body: 'This nanoTabICL-shaped inspector receives twelve labeled Iris examples and one unlabeled query using mapped official TabICLv2 weights. It is not the standalone nanoTabICL model. This trace is outside the documented 300 to 48K-row pretraining regime and is not evidence of expected model quality. Choose Query A, B, or C above. Its hidden species is never supplied to the model.'
		},
		{
			title: '1 · Read the table',
			body: 'Each E row is a labeled context example. Q is the selected unlabeled query. Row IDs E1–E12 and Q stay fixed through the full visualization.'
		},
		{
			title: '2 · Standardize, group, embed',
			body: 'Feature values are standardized with context-only mean and standard deviation, circularly grouped at offsets 1, 2, and 4, then projected into 128-dimensional column tokens.'
		},
		{
			title: '3 · Column attention',
			body: 'Three induced Column Transformer blocks process each feature column. The colored strips are sampled live activations; the thin repeated cards describe symbolic architecture.'
		},
		{
			title: '4 · Compress each row',
			body: 'Three Row Transformer blocks combine feature tokens with four learned CLS tokens. The final four CLS outputs concatenate into one 512-dimensional row vector.'
		},
		{
			title: '5 · Route through context',
			body: 'Twelve ICL Transformer blocks let Q attend to E1–E12. Attention is routing, not feature importance. Change block and head, or expand the matrix to inspect real logits and weights.'
		},
		{
			title: '6 · Predict the class',
			body: 'The final query vector passes through output normalization and an MLP. The three bars are computed softmax probabilities for one selected TabICLv2 model view using mapped official weights.'
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
