import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';
import type { Plugin, PreviewServer, ViteDevServer } from 'vite';
import fs from 'node:fs';
import path from 'node:path';
import type { IncomingMessage, ServerResponse } from 'node:http';

const modelDirectory = path.resolve('../website/model');

function tabiclModelAssets(): Plugin {
	const middleware = (req: IncomingMessage, res: ServerResponse, next: () => void) => {
		const match = req.url?.match(/^\/(?:tabicl-model|model)\/(manifest\.json|tabicl\.bin)(?:\?.*)?$/);
		if (!match) return next();
		const filePath = path.join(modelDirectory, match[1]);
		if (!fs.existsSync(filePath)) {
			res.statusCode = 404;
			res.end('TabICL model asset not found');
			return;
		}
		res.setHeader(
			'Content-Type',
			match[1].endsWith('.json') ? 'application/json' : 'application/octet-stream'
		);
		res.setHeader('Content-Length', fs.statSync(filePath).size);
		res.setHeader('Cache-Control', 'public, max-age=3600');
		fs.createReadStream(filePath).pipe(res);
	};

	return {
		name: 'tabicl-sibling-model-assets',
		configureServer(server: ViteDevServer) {
			server.middlewares.use(middleware);
		},
		configurePreviewServer(server: PreviewServer) {
			server.middlewares.use(middleware);
		}
	};
}

export default defineConfig({
	plugins: [tabiclModelAssets(), sveltekit()],
	server: {
		fs: {
			allow: ['..', '../website']
		}
	}
});
