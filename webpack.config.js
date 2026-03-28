import path from "path";
import { fileURLToPath } from "url";

import CopyPlugin from "copy-webpack-plugin";
import MiniCssExtractPlugin from "mini-css-extract-plugin";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/** @type {import('webpack').Configuration} */
const config = {
  // Mode will be set by CLI argument (--mode=production or --mode=development)
  devtool: process.env.NODE_ENV === 'production' ? false : "inline-source-map",
  entry: {
    'background/index': './src/background/index.ts',
    'popup/index': './src/popup/index.tsx',
    'popup/styles': './src/styles/popup.css',
    'content-scripts/main-page/index': './src/content-scripts/main-page/index.ts',
    'content-scripts/item-page/index': './src/content-scripts/item-page/index.ts',
    'content-scripts/profile-page/index': './src/content-scripts/profile-page/index.ts',
  },
  resolve: {
    alias: {
      "@huggingface/transformers": path.resolve(
        __dirname,
        "node_modules/@huggingface/transformers/dist/transformers.web.js",
      ),
    },
    extensions: ['.tsx', '.ts', '.js'],
  },
  output: {
    path: path.resolve(__dirname, "dist"),
    filename: "[name].js",

    // Otherwise we get `Uncaught ReferenceError: document is not defined`
    chunkLoading: false,
  },
  plugins: [
    new CopyPlugin({
      patterns: [
        {
          from: "public",
          to: ".", // Copies to dist folder
        }
      ],
    }),
    new MiniCssExtractPlugin({
      filename: '[name].css',
    }),
  ],
  module: {
    parser: {
      javascript: {
        importMeta: false,
      },
    },
    rules: [
      {
        test: /\.tsx?$/,
        use: 'ts-loader',
        exclude: /node_modules/,
      },
      {
        test: /\.css$/,
        use: [
          MiniCssExtractPlugin.loader,
          'css-loader',
          'postcss-loader',
        ],
      }
    ]
  },
};

export default config;
