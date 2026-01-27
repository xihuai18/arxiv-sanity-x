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

function _safeMkdir(dir) {
    fs.mkdirSync(dir, { recursive: true });
}

function _cleanDistKeepLib() {
    if (!fs.existsSync(DIST_DIR)) return;
    const files = fs.readdirSync(DIST_DIR);
    for (const file of files) {
        if (file === 'lib' || file.startsWith('.')) continue;
        const filePath = path.join(DIST_DIR, file);
        fs.rmSync(filePath, { recursive: true, force: true });
    }
}

function _copyDirRecursive(srcDir, dstDir) {
    _safeMkdir(dstDir);
    const entries = fs.readdirSync(srcDir, { withFileTypes: true });
    for (const ent of entries) {
        if (!ent || !ent.name || ent.name.startsWith('.')) continue;
        const srcPath = path.join(srcDir, ent.name);
        const dstPath = path.join(dstDir, ent.name);
        if (ent.isDirectory()) {
            _copyDirRecursive(srcPath, dstPath);
        } else if (ent.isFile()) {
            fs.copyFileSync(srcPath, dstPath);
        }
    }
}

function _rmDirRecursive(dir) {
    if (!fs.existsSync(dir)) return;
    fs.rmSync(dir, { recursive: true, force: true });
}

async function build() {
    const isWatch = process.argv.includes('--watch');
    const isDev = process.argv.includes('--dev');

    _safeMkdir(DIST_DIR);

    // For non-watch builds, write into a temporary directory first and only
    // replace dist/ files after the build succeeded. This prevents dist/ from
    // becoming empty when the build fails (e.g. syntax error, missing deps).
    const outDir = isWatch
        ? DIST_DIR
        : path.join(DIST_DIR, `.tmp-build-${process.pid}-${Date.now()}`);
    if (!isWatch) {
        _rmDirRecursive(outDir);
        _safeMkdir(outDir);
    }

    // JS build options
    const jsBuildOptions = {
        entryPoints: JS_ENTRY_POINTS,
        outdir: outDir,
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
        outdir: outDir,
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
        generateManifest(jsResult.metafile, cssResult.metafile, outDir);
        console.log('✅ Initial build complete');
    } else {
        try {
            const jsResult = await esbuild.build(jsBuildOptions);
            const cssResult = await esbuild.build(cssBuildOptions);
            generateManifest(jsResult.metafile, cssResult.metafile, outDir);

            _cleanDistKeepLib();
            _copyDirRecursive(outDir, DIST_DIR);

            console.log('✅ Build complete');
        } finally {
            _rmDirRecursive(outDir);
        }
    }
}

function generateManifest(jsMetafile, cssMetafile, outDir) {
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
    const manifestPath = path.join(outDir, 'manifest.json');
    fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
    console.log(`📝 Generated manifest.json with ${Object.keys(manifest).length} entries`);
}

build().catch(err => {
    console.error('❌ Build failed:', err);
    process.exit(1);
});
