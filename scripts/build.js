#!/usr/bin/env node
/**
 * esbuild build script
 *
 * Features:
 * 1. Generate filenames with hash (for long-term caching)
 * 2. Generate sourcemaps (for debugging)
 * 3. Generate manifest.json (for Flask to read)
 */

const esbuild = require('esbuild');
const fs = require('fs');
const path = require('path');

const STATIC_DIR = path.join(__dirname, '..', 'static');
const DIST_DIR = path.join(STATIC_DIR, 'dist');
const MANIFEST_PATH = path.join(DIST_DIR, 'manifest.json');

// JS entry points list
const JS_ENTRY_POINTS = [
    'paper_list.js',
    'word_list.js',
    'paper_detail.js',
    'paper_summary.js',
    'readinglist.js',
    'tag_dropdown_shared.js',
    'author_utils.js',
    'tldr_utils.js',
    'common_utils.js',
    'markdown_summary_utils.js',
    'markdown_core_utils.js',
    'markdown_sanitizer_utils.js',
    'markdown_renderer_utils.js',
    'markdown_summary_dom_utils.js',
].map(f => path.join(STATIC_DIR, f));

// CSS entry points list
const CSS_ENTRY_POINTS = ['css/main.css'].map(f => path.join(STATIC_DIR, f));

async function build() {
    // Clean old dist files (keep lib directory)
    if (fs.existsSync(DIST_DIR)) {
        const files = fs.readdirSync(DIST_DIR);
        for (const file of files) {
            if (file !== 'lib' && !file.startsWith('.')) {
                const filePath = path.join(DIST_DIR, file);
                const stat = fs.statSync(filePath);
                if (stat.isFile()) {
                    fs.unlinkSync(filePath);
                }
            }
        }
    } else {
        fs.mkdirSync(DIST_DIR, { recursive: true });
    }

    const isWatch = process.argv.includes('--watch');
    const isDev = process.argv.includes('--dev');

    // JS build options
    const jsBuildOptions = {
        entryPoints: JS_ENTRY_POINTS,
        outdir: DIST_DIR,
        bundle: false, // Don't bundle, keep files separate
        minify: !isDev,
        sourcemap: true, // Always generate sourcemap
        target: 'es2017',
        format: 'iife',
        loader: { '.js': 'jsx' },
        jsxFactory: 'React.createElement',
        jsxFragment: 'React.Fragment',
        // Use hash filenames (production mode)
        entryNames: isDev ? '[name]' : '[name]-[hash]',
        metafile: true,
    };

    // CSS build options
    const cssBuildOptions = {
        entryPoints: CSS_ENTRY_POINTS,
        outdir: DIST_DIR,
        bundle: true, // CSS needs bundling to process @import
        minify: !isDev,
        sourcemap: true,
        // Use hash filenames (production mode)
        entryNames: isDev ? '[name]' : '[name]-[hash]',
        metafile: true,
        // External resource handling: keep URLs unchanged
        loader: {
            '.png': 'file',
            '.jpg': 'file',
            '.jpeg': 'file',
            '.gif': 'file',
            '.svg': 'file',
            '.woff': 'file',
            '.woff2': 'file',
            '.ttf': 'file',
            '.eot': 'file',
        },
        // Mark absolute path resources as external
        external: ['/static/*'],
    };

    if (isWatch) {
        const jsCtx = await esbuild.context(jsBuildOptions);
        const cssCtx = await esbuild.context(cssBuildOptions);
        await jsCtx.watch();
        await cssCtx.watch();
        console.log('👀 Watching for changes...');

        // Initial build
        const jsResult = await jsCtx.rebuild();
        const cssResult = await cssCtx.rebuild();
        generateManifest(jsResult.metafile, cssResult.metafile);
        console.log('✅ Initial build complete');
    } else {
        const jsResult = await esbuild.build(jsBuildOptions);
        const cssResult = await esbuild.build(cssBuildOptions);
        generateManifest(jsResult.metafile, cssResult.metafile);
        console.log('✅ Build complete');
    }
}

function generateManifest(jsMetafile, cssMetafile) {
    const manifest = {};

    // Process JS files
    for (const [outputPath, info] of Object.entries(jsMetafile.outputs)) {
        if (outputPath.endsWith('.map')) continue;

        const entryPoint = info.entryPoint;
        if (!entryPoint) continue;

        // Get original filename (without path)
        const originalName = path.basename(entryPoint);
        // Get output filename (relative to dist directory)
        const outputName = path.basename(outputPath);

        manifest[originalName] = outputName;
    }

    // Process CSS files
    for (const [outputPath, info] of Object.entries(cssMetafile.outputs)) {
        if (outputPath.endsWith('.map')) continue;

        const entryPoint = info.entryPoint;
        if (!entryPoint) continue;

        // Get original filename (without path)
        const originalName = path.basename(entryPoint);
        // Get output filename (relative to dist directory)
        const outputName = path.basename(outputPath);

        manifest[originalName] = outputName;
    }

    // Write manifest.json
    fs.writeFileSync(MANIFEST_PATH, JSON.stringify(manifest, null, 2));
    console.log(`📝 Generated manifest.json with ${Object.keys(manifest).length} entries`);
}

build().catch(err => {
    console.error('❌ Build failed:', err);
    process.exit(1);
});
