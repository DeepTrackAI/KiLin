# SVG architecture diagrams — implemented in PR9b

**Status:** Eleven native, editable SVG diagrams, integrated into their
primary subsystem documentation. These figures describe the code state
reviewed for PR9a/PR9b, not intended future capabilities.

## Visual conventions

- **Indigo:** Python core and model/tool execution components.
- **Blue:** optional GUI, desktop presentation or user interaction.
- **Green:** user-owned, persisted project state and data records.
- **Amber:** packaged inputs or optional operations and callbacks.
- **Rose:** a failure path, risk, or explicit trust limitation.
- **Solid arrows:** implemented data, control or ownership transfer.
- **Dashed arrows:** optional/conditional paths, or error/recovery paths
  where the adjacent label identifies the condition.
- **Colored area backgrounds:** conceptual grouping or process/data
  ownership. Only labeled subprocess zones indicate OS process separation.

These conventions describe **diagram emphasis**, not a formal type system.
For example, a workspace is a persistent folder, not an execution sandbox;
`QWebChannel` is not a network API; and an optional model download is not
necessarily a normal part of an offline RAG operation.

## Figure inventory and owners

| Editable SVG | Authoritative document | Main question |
| --- | --- | --- |
| [architecture.svg](figures/architecture.svg) | [Architecture](architecture.md) | Which components can operate without Qt? |
| [workspace-ownership.svg](figures/workspace-ownership.svg) | [Project workspaces](project_workspaces.md) | What is packaged, user-wide, or per-project? |
| [source-ingestion.svg](figures/source-ingestion.svg) | [Source pipeline](source_pipeline.md) | How do selection, imports and chunks differ? |
| [retrieval-indexes.svg](figures/retrieval-indexes.svg) | [Retrieval](retrieval_pipeline.md) | How do the three ranked channels produce evidence? |
| [answer-sequence.svg](figures/answer-sequence.svg) | [Answering](answering_pipeline.md) | Which stages precede a committed answer? |
| [model-lifetimes.svg](figures/model-lifetimes.svg) | [Model management](model_management.md) | What is loaded, reused, or released? |
| [gui-processes.svg](figures/gui-processes.svg) | [GUI architecture](gui_architecture.md) | What crosses Qt and subprocess boundaries? |
| [configuration-precedence.svg](figures/configuration-precedence.svg) | [Configuration](configuration.md) | How do creation defaults differ from saved settings? |
| [tool-contract.svg](figures/tool-contract.svg) | [Extension interfaces](extension_interfaces.md) | How does an intent invoke a registered tool? |
| [conversation-ownership.svg](figures/conversation-ownership.svg) | [Conversation ownership](conversation_ownership.md) | How do locks and atomic checkpoints interact? |
| [trust-boundaries.svg](figures/trust-boundaries.svg) | [Security](security.md) | What remains untrusted or potentially external? |

The illustrations intentionally omit some implementation detail to remain
legible. In particular, the answering sequence summarizes optional paths
rather than claiming every intent executes them; the configuration diagram
does not imply that application-wide appearance settings overwrite saved
project profiles; and the source pipeline distinguishes policy checks from
the two managed source-import APIs.

## Maintenance contract

1. Review the current implementation and tests before changing an arrow or
   naming a process, trust boundary or API.
2. Keep text as native `<text>` elements and the layout as SVG vector shapes.
   Do not flatten diagrams into screenshots or embed unrelated remote assets.
3. Preserve meaningful element IDs, SVG `<title>`/`<desc>` descriptions, and
   explanatory alt text/captions in the corresponding Markdown documents.
4. Keep one primary document per figure. Update the owner document together
   with any modified figure to avoid contradictory architecture claims.
5. Render each changed SVG at ordinary documentation size, inspect text and
   arrow placement, parse SVG/XML and check all local Markdown references.
6. Do not infer a runtime guarantee merely from a diagram; the detailed
   documented contracts and source tests determine actual behavior.

No runtime code or GUI behavior was modified to make these diagrams simpler.
The root `README.md` remains a compact entry point rather than a gallery.
