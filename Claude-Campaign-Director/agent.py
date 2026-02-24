"""
Campaign Agent Orchestrator v2
Connects to MCP server and runs the full asset generation pipeline.
Image generation powered by Nano Banana Pro (gemini-3-pro-image-preview).
Streams reasoning events back to the UI via callback.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Callable, Optional

import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY", "")
MCP_SERVER_PATH   = Path(__file__).parent / "server.py"

SYSTEM_PROMPT = """You are a creative production agent for a world-class advertising agency.
Your job: take a campaign brief and produce polished, publication-ready ad images using Nano Banana Pro — Google's professional image generation model.

You have 6 tools. Use them in this exact order for each platform × language combination:

## Pipeline

### Phase 1 — Brief & Style (do ONCE at the start)
1. `parse_brief` — Always call FIRST. Extracts brand, platforms, languages, audience, legal constraints.
2. `analyze_references` — Call if reference images are provided. Extracts visual style, colors, lighting, composition. This informs EVERY image generation prompt.

### Phase 2 — Per Platform × Language (repeat for each combo)
3. `generate_copy` — Write platform-native copy for this language
4. `score_copy` — ALWAYS score after generating. If score < 75, apply the `revised_copy` from the score result and re-score. Max 2 revision attempts, then proceed with best version.
5. `build_image_prompt` — Craft the Nano Banana Pro prompt. This is critical — take time here. The prompt must include all copy text to be rendered in the image, full visual scene description, and style descriptors from analyze_references.
6. `generate_image` — Call Nano Banana Pro. Pass reference_image_paths if product images were provided (helps model understand the product visually).

## Key rules
- The CREATIVE IMAGE is the deliverable. Copy exists to be rendered inside it by Nano Banana Pro — not as a separate text layer.
- Always pass style_descriptor from analyze_references into build_image_prompt.
- Always pass product image paths to generate_image when available — Nano Banana Pro will use them as visual reference.
- Think out loud before each tool call. Explain your creative reasoning.
- Be decisive. Work with what you have. Never ask for clarification.
- At the end, list all generated image paths grouped by platform.
"""


class CampaignAgent:
    def __init__(self, on_event: Optional[Callable] = None):
        self.on_event = on_event or (lambda e: print(f"[agent] {e}"))
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def emit(self, event_type: str, data: dict):
        self.on_event({"type": event_type, **data})

    async def run(
        self,
        brief_text: str,
        brand_guidelines: str = "",
        ratio_specs: str = "",
        reference_image_paths: list[str] = None,
        product_image_paths: list[str] = None,
    ) -> dict:
        reference_image_paths = reference_image_paths or []
        product_image_paths   = product_image_paths   or []

        self.emit("status", {"message": "Initializing campaign agent…", "step": "init"})

        server_params = StdioServerParameters(
            command="python3",
            args=[str(MCP_SERVER_PATH)],
            env={
                **os.environ,
                "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
                "GEMINI_API_KEY":    GEMINI_API_KEY,
            },
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # Pull tool definitions from the MCP server
                mcp_tools = await session.list_tools()
                tools = [
                    {
                        "name":         t.name,
                        "description":  t.description,
                        "input_schema": t.inputSchema,
                    }
                    for t in mcp_tools.tools
                ]

                # Build the user message
                ref_note = ""
                if reference_image_paths:
                    ref_note += f"\n\nSTYLE REFERENCE IMAGES (analyze these first): {json.dumps(reference_image_paths)}"
                if product_image_paths:
                    ref_note += f"\n\nPRODUCT IMAGES (pass to generate_image for visual context): {json.dumps(product_image_paths)}"

                user_message = f"""Generate all campaign assets for this brief. Produce complete ad images using Nano Banana Pro.

CAMPAIGN BRIEF:
{brief_text}

BRAND & LEGAL GUIDELINES:
{brand_guidelines or "Not provided — infer from brief."}

ASPECT RATIO SPECS:
{ratio_specs or "Not provided — infer from brief."}
{ref_note}

Follow the full pipeline for each platform × language. The output is the IMAGE — Nano Banana Pro renders the copy inside it.
Report every generated image path at the end."""

                messages = [{"role": "user", "content": user_message}]
                outputs  = []

                self.emit("status", {"message": "Agent reasoning…", "step": "reasoning"})

                # ── Agentic loop ──────────────────────────────────────────────
                while True:
                    response = self.client.messages.create(
                        model="claude-opus-4-5",
                        max_tokens=4096,
                        system=SYSTEM_PROMPT,
                        tools=tools,
                        messages=messages,
                    )

                    # Stream reasoning text
                    for block in response.content:
                        if block.type == "text" and block.text.strip():
                            self.emit("reasoning", {"text": block.text})

                    if response.stop_reason == "end_turn":
                        self.emit("status", {"message": "Pipeline complete.", "step": "done"})
                        break

                    if response.stop_reason == "tool_use":
                        tool_results = []

                        for block in response.content:
                            if block.type != "tool_use":
                                continue

                            self.emit("tool_call", {
                                "tool":  block.name,
                                "input": block.input,
                            })

                            result = await session.call_tool(block.name, block.input)
                            result_text = result.content[0].text if result.content else "{}"

                            self.emit("tool_result", {
                                "tool":   block.name,
                                "result": result_text[:600],
                            })

                            # Track generated images
                            if block.name == "generate_image":
                                try:
                                    r = json.loads(result_text)
                                    if r.get("success") and r.get("output_path"):
                                        outputs.append({
                                            "path":     r["output_path"],
                                            "platform": r.get("platform", ""),
                                            "language": r.get("language", ""),
                                            "ratio":    r.get("aspect_ratio", ""),
                                            "model":    r.get("model", ""),
                                        })
                                        self.emit("asset_rendered", {
                                            "path":     r["output_path"],
                                            "platform": r.get("platform", ""),
                                            "language": r.get("language", ""),
                                        })
                                except Exception:
                                    pass

                            tool_results.append({
                                "type":        "tool_result",
                                "tool_use_id": block.id,
                                "content":     result_text,
                            })

                        messages.append({"role": "assistant", "content": response.content})
                        messages.append({"role": "user",      "content": tool_results})
                    else:
                        break

                return {"success": True, "outputs": outputs}


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    def print_event(e):
        t = e.get("type")
        if t == "reasoning":
            print(f"\n🧠 {e['text'][:250]}")
        elif t == "tool_call":
            print(f"\n⚡ TOOL → {e['tool']}")
        elif t == "tool_result":
            print(f"   ↳ {e['result'][:120]}")
        elif t == "asset_rendered":
            print(f"\n✅ IMAGE SAVED → {e['path']}")
        elif t == "status":
            print(f"\n📍 {e['message']}")

    TEST_BRIEF = """Campaign Brief: "The Speed of You"
Brand: Nike
Campaign Period: Q2 2026 (Global Launch)

Primary Products:
- Nike Air Zoom Pegasus 43 (Performance Running)
- Nike Alphafly 4 (Elite Racing)

Target Region/Market: Global (Priority: NYC, London, Mexico City)

Target Audience:
Primary: "The Everyday Athlete" (Ages 18–35). Urban dwellers who balance high-pressure careers with a disciplined fitness routine.
Secondary: Competitive runners and Gen Z fitness enthusiasts seeking high-performance gear with sustainability credentials.

Campaign Message:
Core Tagline: Your Speed, Your Science.
Concept: Shifting the narrative from "breaking world records" to "breaking personal barriers." Nike's elite technology is now engineered for the everyday runner's cadence and lifestyle.

Languages: EN, ES"""

    TEST_RATIOS = """
    TikTok / Reels: 9:16, 1080x1920
    Instagram: 4:5, 1080x1350
    """

    agent = CampaignAgent(on_event=print_event)
    result = asyncio.run(agent.run(TEST_BRIEF, ratio_specs=TEST_RATIOS))

    print(f"\n{'='*60}")
    print(f"DONE — {len(result['outputs'])} images generated:")
    for o in result["outputs"]:
        print(f"  [{o['platform']} / {o['language']} / {o['ratio']}] {o['path']}")
