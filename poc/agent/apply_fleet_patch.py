#!/usr/bin/env python3
"""Patch NVIDIA's nvtrace_continuous_export (Nsight Perf SDK for AI, beta-preview-update1) to add FleetSink.

Usage: apply_fleet_patch.py <sdk-source-root>

Copies NvgxFleetSink.h next to nvtrace-tools/ContinuousExport/nvtrace_continuous_export.cpp and edits the
.cpp so every snapshot goes to a TeeSink (DuckDB as before + FleetSink). Adds two options:
  --fleet-url URL    POST workload profiles to URL/v1/profiles (empty = disabled, stock behaviour)
  --fleet-node NAME  node id sent with each profile (default: hostname)
Each edit anchors on exact upstream text and fails loudly if the source differs.
"""
import shutil
import sys
from pathlib import Path

EDITS = [
    ("#include <NvTraceProviderIgnoreCli.h>\n",
     "#include <NvTraceProviderIgnoreCli.h>\n#include \"NvgxFleetSink.h\"\n"),
    ("        std::filesystem::path outputDirectory;\n        std::filesystem::path databasePath;\n"
     "        nv::trace::Codec::ProviderTypeTable<bool> isProviderTypeIgnored = {};\n",
     "        std::filesystem::path outputDirectory;\n        std::filesystem::path databasePath;\n"
     "        nv::trace::Codec::ProviderTypeTable<bool> isProviderTypeIgnored = {};\n"
     "        std::string fleetUrl;\n        std::string fleetNode;\n"),
    ("        nv::trace::DuckDbTableSink sink;\n    };\n",
     "        nv::trace::DuckDbTableSink sink;\n        nvgx::FleetSink fleet;\n"
     "        nvgx::TeeSink<nv::trace::DuckDbTableSink, nvgx::FleetSink> tee{sink, fleet};\n    };\n"),
    ("        nv::trace::DuckDbTableSink& sink = state.sink;\n",
     "        auto& sink = state.tee;\n"),
    ("nv::trace::TableExporter<SnapshotDatabaseReader, nv::trace::DuckDbTableSink> tableExporter(dataModel);",
     "nv::trace::TableExporter<SnapshotDatabaseReader, nvgx::TeeSink<nv::trace::DuckDbTableSink, nvgx::FleetSink>> tableExporter(dataModel);"),
    ("            (\"h,help\", \"Print usage\");\n",
     "            (\"fleet-url\", \"nvgx: POST a workload profile per snapshot to URL/v1/profiles (empty = disabled)\", cxxopts::value<std::string>()->default_value(\"\"))\n"
     "            (\"fleet-node\", \"nvgx: node id sent with each profile (default: hostname)\", cxxopts::value<std::string>()->default_value(\"\"))\n"
     "            (\"h,help\", \"Print usage\");\n"),
    ("        out_options.showSchemaOnly = result[\"show-schema-only\"].as<bool>();\n",
     "        out_options.showSchemaOnly = result[\"show-schema-only\"].as<bool>();\n"
     "        out_options.fleetUrl = result[\"fleet-url\"].as<std::string>();\n"
     "        out_options.fleetNode = result[\"fleet-node\"].as<std::string>();\n"),
    ("            std::cerr << \"Error: failed to prepare database \" << state.databasePath.string() << \".\\n\";\n"
     "            return EXIT_FAILURE;\n        }\n",
     "            std::cerr << \"Error: failed to prepare database \" << state.databasePath.string() << \".\\n\";\n"
     "            return EXIT_FAILURE;\n        }\n"
     "        if (!options.fleetUrl.empty())\n        {\n"
     "            std::string node = options.fleetNode;\n"
     "            if (node.empty())\n            {\n"
     "                char host[256] = {};\n                gethostname(host, sizeof(host) - 1);\n                node = host;\n            }\n"
     "            if (!state.fleet.Open(options.fleetUrl, node))\n            {\n"
     "                std::cerr << \"Error: invalid --fleet-url \" << options.fleetUrl << \".\\n\";\n"
     "                return EXIT_FAILURE;\n            }\n        }\n"),
    ("        state.sink.Close();\n",
     "        state.sink.Close();\n        state.fleet.Close();\n"),
]


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    root = Path(sys.argv[1])
    cpp = root / "nvtrace-tools" / "ContinuousExport" / "nvtrace_continuous_export.cpp"
    text = cpp.read_text()
    if "NvgxFleetSink.h" in text:
        sys.exit(f"{cpp} is already patched")
    for i, (old, new) in enumerate(EDITS, 1):
        if text.count(old) != 1:
            sys.exit(f"edit {i}: anchor found {text.count(old)} times (expected 1); upstream source changed?")
        text = text.replace(old, new)
    shutil.copy(Path(__file__).with_name("NvgxFleetSink.h"), cpp.parent / "NvgxFleetSink.h")
    cpp.write_text(text)
    print(f"patched {cpp} ({len(EDITS)} edits) and copied NvgxFleetSink.h")


if __name__ == "__main__":
    main()
