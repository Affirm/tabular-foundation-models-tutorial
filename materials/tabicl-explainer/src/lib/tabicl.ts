// @ts-ignore The browser model is maintained by the sibling TabICL website.
import { buildModel } from '../../../website/js/tabicl/nanotabicl.js';
// @ts-ignore The integrity helper is maintained with the sibling browser runtime.
import { verifySha256 } from '../../../website/js/tabicl/tensor.js';

export const FEATURES = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width'];
export const CLASS_NAMES = ['Setosa', 'Versicolor', 'Virginica'];

export const CONTEXT = [
	{ id: 'E1', x: [5.1, 3.5, 1.4, 0.2], y: 0 },
	{ id: 'E2', x: [4.9, 3.0, 1.4, 0.2], y: 0 },
	{ id: 'E3', x: [4.7, 3.2, 1.3, 0.2], y: 0 },
	{ id: 'E4', x: [4.6, 3.1, 1.5, 0.2], y: 0 },
	{ id: 'E5', x: [7.0, 3.2, 4.7, 1.4], y: 1 },
	{ id: 'E6', x: [6.4, 3.2, 4.5, 1.5], y: 1 },
	{ id: 'E7', x: [6.9, 3.1, 4.9, 1.5], y: 1 },
	{ id: 'E8', x: [5.5, 2.3, 4.0, 1.3], y: 1 },
	{ id: 'E9', x: [6.3, 3.3, 6.0, 2.5], y: 2 },
	{ id: 'E10', x: [5.8, 2.7, 5.1, 1.9], y: 2 },
	{ id: 'E11', x: [7.1, 3.0, 5.9, 2.1], y: 2 },
	{ id: 'E12', x: [6.3, 2.9, 5.6, 1.8], y: 2 }
] as const;

export const QUERIES = [
	{ id: 'A', x: [5.0, 3.6, 1.4, 0.2] },
	{ id: 'B', x: [6.1, 2.9, 4.7, 1.4] },
	{ id: 'C', x: [6.5, 3.0, 5.8, 2.2] }
] as const;

export type LoadPhase = 'downloading' | 'initializing' | 'running' | 'ready' | 'error';

export type LoadStatus = {
	phase: LoadPhase;
	progress: number;
	message: string;
};

export type AttentionTrace = {
	average: number[];
	heads: number[][];
	rawLogits: number[][];
	scaledLogits: number[][];
};

export type MatrixSample = {
	rows: number;
	dim: number;
	shown: number;
	values: number[][];
	stats: { min: number; max: number; mean: number; rms: number };
};

export type Inspection = {
	config: {
		embedDim: number;
		iclDim: number;
		featureGroupSize: number;
		groupOffsets: number[];
		columnBlocks: number;
		rowBlocks: number;
		iclBlocks: number;
		heads: number;
		inducing: number;
		clsTokens: number;
	};
	input: {
		raw: number[];
		mean: number[];
		std: number[];
		standardized: number[];
		grouped: number[][];
	};
	embedding: MatrixSample;
	columnBlocks: { block: number; activation: MatrixSample }[];
	rowBlocks: { block: number; tokens: MatrixSample }[];
	rowVector: MatrixSample;
	iclBlocks: { block: number; activation: MatrixSample; attention: AttentionTrace }[];
	output: {
		normalized: MatrixSample;
		hidden: MatrixSample;
		logits: number[];
		probabilities: number[];
	};
};

export type TabICLModel = {
	prepareContext: (x: number[][], y: number[]) => unknown;
	inspectQuery: (cache: unknown, query: number[], nClasses: number) => Inspection;
};

async function fetchWithProgress(
	url: string,
	onProgress: (loaded: number, total: number) => void
): Promise<ArrayBuffer> {
	const response = await fetch(url, { cache: 'no-store' });
	if (!response.ok) throw new Error(`Model download failed (${response.status})`);
	const total = Number(response.headers.get('content-length')) || 27_894_972;
	if (!response.body) {
		const buffer = await response.arrayBuffer();
		onProgress(buffer.byteLength, total);
		return buffer;
	}

	const reader = response.body.getReader();
	const chunks: Uint8Array[] = [];
	let loaded = 0;
	while (true) {
		const { done, value } = await reader.read();
		if (done) break;
		if (value) {
			chunks.push(value);
			loaded += value.byteLength;
			onProgress(loaded, total);
		}
	}
	const merged = new Uint8Array(loaded);
	let offset = 0;
	for (const chunk of chunks) {
		merged.set(chunk, offset);
		offset += chunk.byteLength;
	}
	return merged.buffer;
}

export async function loadTabICL(
	onStatus: (status: LoadStatus) => void
): Promise<{ model: TabICLModel; cache: unknown }> {
	onStatus({ phase: 'downloading', progress: 0, message: 'Downloading official TabICLv2 weights' });
	const modelBase = new URL('../model/', window.location.href);
	const manifestResponse = await fetch(new URL('manifest.json', modelBase), { cache: 'no-store' });
	if (!manifestResponse.ok) throw new Error(`Manifest download failed (${manifestResponse.status})`);
	const manifest = await manifestResponse.json();
	const binaryName = manifest.binary || 'tabicl.bin';
	const buffer = await fetchWithProgress(new URL(binaryName, modelBase).href, (loaded, total) => {
		onStatus({
			phase: 'downloading',
			progress: Math.min(0.78, (loaded / total) * 0.78),
			message: `Downloading weights · ${(loaded / 1_000_000).toFixed(1)} / ${(total / 1_000_000).toFixed(1)} MB`
		});
	});
	await verifySha256(buffer, manifest.binary_sha256, binaryName);

	onStatus({ phase: 'initializing', progress: 0.82, message: 'Dequantizing int8/fp16 weights' });
	await new Promise((resolve) => setTimeout(resolve, 20));
	const model = buildModel(manifest, buffer) as TabICLModel;

	onStatus({ phase: 'initializing', progress: 0.9, message: 'Encoding the 12 Iris examples' });
	await new Promise((resolve) => setTimeout(resolve, 20));
	const cache = model.prepareContext(
		CONTEXT.map((row) => [...row.x]),
		CONTEXT.map((row) => row.y)
	);
	onStatus({ phase: 'ready', progress: 1, message: 'nanoTabICL trace ready' });
	return { model, cache };
}
