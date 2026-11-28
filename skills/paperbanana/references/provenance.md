# Provenance and licenses

The workflow adapts [PaperBanana](https://github.com/dwzhu-pku/PaperBanana), inspected
revision `9ceea03`, especially `agents/retriever_agent.py`, `planner_agent.py`,
`stylist_agent.py`, `visualizer_agent.py`, `critic_agent.py` and
`utils/paperviz_processor.py`. That upstream repository licenses its code under
Apache-2.0. This skill independently expresses the stage methodology; it does not
bundle upstream Python, style guides, reference figures or a reference dataset.
No dataset license is assumed.

The immediate workflow source is the installed Autoresearch paper workflow's
`paper/references/figure-production.md`. It added actual-venue styling, tool
capability discovery, baseline-image editing, fail-closed critique, information
reduction for main figures, and evidence-bound manuscript integration. The
standalone skill keeps the figure-production responsibilities and leaves research
controller responsibilities with its caller.

Image export and artifact helpers in `scripts/paperbanana.py` adapt
Autoresearch's `paper/lib/production.py` and `paper/lib/runtime.py`, licensed MIT,
Copyright (c) 2026 Udit Goenka. The MIT notice is retained in `../LICENSE`.

The optional Magpie transport adapts a local smoke-test helper that demonstrated
image generation and an input-image label edit over stdio MCP. The first edit
lost its asynchronous task after the process exited; a separate replacement run
succeeded with same-process retrieval. Those historical results informed the
transport design, but do not prove availability or image quality in a new host.
Private project materials, original generated images, credentials and local
absolute installation paths are not bundled as examples.
