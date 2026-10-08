// CLI: node build_xml.mjs config.json out.xml
// Writes the MJCF that web/js/model_builder.js produces for a config (for tests/parity_check.py).
import fs from "node:fs";
import { assets } from "./helpers.mjs";
import { buildModelXml } from "../../web/js/model_builder.js";

const [cfgPath, outPath] = process.argv.slice(2);
if (!cfgPath || !outPath) {
  console.error("usage: node build_xml.mjs config.json out.xml");
  process.exit(2);
}
const { baseXml, catalog } = assets();
fs.writeFileSync(outPath, buildModelXml(baseXml, catalog, JSON.parse(fs.readFileSync(cfgPath, "utf8"))));
