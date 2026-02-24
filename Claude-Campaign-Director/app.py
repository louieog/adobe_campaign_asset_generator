"""
Campaign Asset Generator — Demo UI v2
Powered by Claude (orchestration) + Nano Banana Pro (image generation)
Run: python app.py
"""

import asyncio
import json
import os
import threading
import time
from pathlib import Path

import gradio as gr

from agent import CampaignAgent


# ── Demo content ──────────────────────────────────────────────────────────────
NIKE_BRIEF = """Campaign Brief: "The Speed of You"
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

NIKE_GUIDELINES = """Brand Voice: Bold, aspirational, direct. Short sentences. Active verbs only. No passive constructions.
Avoid: "we", "our products", unsubstantiated superlatives like "best" or "greatest".
Legal: No performance claims implying medical benefits. Footwear claims limited to comfort and speed.
Tagline must appear verbatim: "Your Speed, Your Science." — do not paraphrase or abbreviate.
For non-English: translate concept culturally; tagline may be adapted if literal translation loses impact — note the adaptation."""

NIKE_RATIOS = """TikTok / Reels: Full-Screen Video, 9:16, 1080x1920
Instagram / Threads: In-Feed Portrait, 4:5, 1080x1350
YouTube Shorts: Vertical Video, 9:16, 1080x1920
X (Twitter) / LinkedIn: Square Post, 1:1, 1200x1200"""

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

:root {
  --bg: #080809;
  --surface: #101012;
  --surface2: #18181C;
  --border: #242428;
  --accent: #6C5CE7;
  --accent2: #FD79A8;
  --green: #00CEC9;
  --yellow: #FDCB6E;
  --text: #EDEDF0;
  --muted: #606070;
  --font-mono: 'Space Mono', monospace;
  --font-sans: 'Syne', sans-serif;
}

* { box-sizing: border-box; }

body, .gradio-container {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--font-sans) !important;
  min-height: 100vh;
}

/* Panels */
.gr-panel, .gr-box, .block {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
}

/* Buttons */
.gr-button-primary {
  background: linear-gradient(135deg, var(--accent), #A29BFE) !important;
  border: none !important;
  font-family: var(--font-mono) !important;
  font-size: 12px !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  padding: 14px 32px !important;
  border-radius: 6px !important;
  color: #fff !important;
  font-weight: 700 !important;
  transition: all 0.25s ease !important;
  box-shadow: 0 4px 20px rgba(108,92,231,0.35) !important;
}
.gr-button-primary:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 30px rgba(108,92,231,0.5) !important;
}
.gr-button-primary:disabled {
  opacity: 0.4 !important;
  transform: none !important;
  cursor: not-allowed !important;
}

.gr-button-secondary {
  background: transparent !important;
  border: 1px solid var(--border) !important;
  color: var(--muted) !important;
  font-family: var(--font-mono) !important;
  font-size: 10px !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  border-radius: 5px !important;
  transition: all 0.2s !important;
}
.gr-button-secondary:hover {
  border-color: var(--accent) !important;
  color: var(--text) !important;
}

/* Inputs */
textarea, input[type="text"] {
  background: #0A0A0C !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  font-family: var(--font-mono) !important;
  font-size: 11.5px !important;
  border-radius: 7px !important;
  line-height: 1.7 !important;
  transition: border-color 0.2s !important;
}
textarea:focus, input[type="text"]:focus {
  border-color: var(--accent) !important;
  outline: none !important;
}

/* Labels */
label, .gr-form > label {
  font-family: var(--font-mono) !important;
  font-size: 10px !important;
  letter-spacing: 0.15em !important;
  text-transform: uppercase !important;
  color: var(--muted) !important;
  margin-bottom: 6px !important;
}

/* Tabs */
.tab-nav button {
  font-family: var(--font-mono) !important;
  font-size: 10px !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  background: transparent !important;
  color: var(--muted) !important;
  border-bottom: 2px solid transparent !important;
  transition: all 0.2s !important;
}
.tab-nav button.selected {
  color: var(--text) !important;
  border-bottom-color: var(--accent) !important;
}

/* Agent log */
.agent-log {
  background: #050507;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 18px 20px;
  font-family: var(--font-mono);
  font-size: 11px;
  line-height: 2;
  height: 460px;
  overflow-y: auto;
  scrollbar-width: thin;
  scrollbar-color: var(--border) transparent;
}
.log-idle    { color: #333340; }
.log-status  { color: var(--yellow); }
.log-think   { color: #8888A8; font-style: italic; }
.log-tool    { color: var(--accent); font-weight: 700; }
.log-result  { color: var(--muted); padding-left: 20px; border-left: 2px solid var(--border); margin-left: 4px; }
.log-score   { color: var(--green); }
.log-score-fail { color: var(--accent2); }
.log-image   { color: var(--green); font-weight: 700; }
.log-error   { color: var(--accent2); }
.log-prompt  { color: #A29BFE; font-size: 10px; padding-left: 20px; opacity: 0.85; }

/* Header */
#app-header {
  padding: 28px 0 20px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 24px;
  display: flex;
  align-items: baseline;
  gap: 16px;
}
#app-header h1 {
  font-family: var(--font-sans);
  font-weight: 800;
  font-size: 26px;
  letter-spacing: -0.03em;
  color: var(--text);
  margin: 0;
}
#app-header .badge {
  font-family: var(--font-mono);
  font-size: 9px;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--accent);
  border: 1px solid var(--accent);
  padding: 3px 8px;
  border-radius: 3px;
  opacity: 0.8;
}

.section-label {
  font-family: var(--font-mono);
  font-size: 9px;
  letter-spacing: 0.25em;
  text-transform: uppercase;
  color: var(--accent);
  display: block;
  margin-bottom: 10px;
  opacity: 0.9;
}

/* Stats bar */
.stat-bar {
  display: flex;
  gap: 24px;
  padding: 12px 0 0;
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--muted);
}
.stat-bar .stat-val {
  color: var(--text);
  font-weight: 700;
}
"""


# ── Log builder ───────────────────────────────────────────────────────────────
def build_log_html(events: list) -> str:
    if not events:
        return '<div class="agent-log"><span class="log-idle">// Agent output will stream here once you start the pipeline.</span></div>'

    lines = []
    for e in events:
        t = e.get("type")

        if t == "reasoning":
            text = e["text"][:300].replace("<", "&lt;").replace(">", "&gt;")
            lines.append(f'<div class="log-think">↳ {text}</div>')

        elif t == "tool_call":
            tool = e.get("tool", "")
            inp  = e.get("input", {})

            # Pretty-print key fields per tool
            if tool == "parse_brief":
                lines.append(f'<div class="log-tool">▶ parse_brief — structuring campaign data</div>')
            elif tool == "analyze_references":
                n = len(inp.get("image_paths", []))
                lines.append(f'<div class="log-tool">▶ analyze_references — vision-analyzing {n} image(s)</div>')
            elif tool == "generate_copy":
                lines.append(f'<div class="log-tool">▶ generate_copy — [{inp.get("platform","")} / {inp.get("language","")}]</div>')
            elif tool == "score_copy":
                lines.append(f'<div class="log-tool">▶ score_copy — [{inp.get("platform","")} / {inp.get("language","")}]</div>')
            elif tool == "build_image_prompt":
                lines.append(f'<div class="log-tool">▶ build_image_prompt — [{inp.get("platform","")} / {inp.get("ratio","")} / {inp.get("language","")}]</div>')
            elif tool == "generate_image":
                lines.append(f'<div class="log-tool">▶ generate_image 🍌 — [{inp.get("platform","")} / {inp.get("aspect_ratio","")} / {inp.get("language","")}] calling Nano Banana Pro…</div>')
            else:
                lines.append(f'<div class="log-tool">▶ {tool}</div>')

        elif t == "tool_result":
            tool   = e.get("tool", "")
            result = e.get("result", "")

            if tool == "score_copy":
                try:
                    r = json.loads(result)
                    score = r.get("total_score", "?")
                    passed = r.get("pass", False)
                    notes  = r.get("approval_notes", "")
                    cls    = "log-score" if passed else "log-score-fail"
                    icon   = "✓" if passed else "✗"
                    lines.append(f'<div class="{cls}">&nbsp;&nbsp;{icon} score: {score}/100 — {notes[:80]}</div>')
                except:
                    lines.append(f'<div class="log-result">&nbsp;&nbsp;{result[:100]}</div>')

            elif tool == "build_image_prompt":
                try:
                    r = json.loads(result)
                    prompt_preview = r.get("prompt", "")[:120].replace("<","&lt;")
                    lines.append(f'<div class="log-prompt">&nbsp;&nbsp;"{prompt_preview}…"</div>')
                    if r.get("art_direction_notes"):
                        lines.append(f'<div class="log-result">&nbsp;&nbsp;🎨 {r["art_direction_notes"][:100]}</div>')
                except:
                    lines.append(f'<div class="log-result">&nbsp;&nbsp;{result[:120]}</div>')

            elif tool == "generate_image":
                try:
                    r = json.loads(result)
                    if r.get("success"):
                        fname = Path(r["output_path"]).name
                        lines.append(f'<div class="log-image">&nbsp;&nbsp;✅ {fname}</div>')
                    else:
                        lines.append(f'<div class="log-error">&nbsp;&nbsp;❌ {r.get("error","generation failed")[:120]}</div>')
                        if r.get("prompt_for_review"):
                            escaped = r["prompt_for_review"][:500].replace("<","&lt;").replace(">","&gt;")
                            lines.append(f'<div class="log-prompt">&nbsp;&nbsp;📝 Image prompt (for manual generation):<br>&nbsp;&nbsp;"{escaped}…"</div>')
                except:
                    lines.append(f'<div class="log-result">&nbsp;&nbsp;{result[:120]}</div>')

            elif tool == "analyze_references":
                try:
                    r = json.loads(result)
                    mood  = r.get("mood", "")
                    style = r.get("style_descriptor", "")[:80]
                    lines.append(f'<div class="log-result">&nbsp;&nbsp;mood: {mood} | {style}</div>')
                except:
                    lines.append(f'<div class="log-result">&nbsp;&nbsp;{result[:100]}</div>')

            else:
                snippet = result[:120].replace("<","&lt;")
                lines.append(f'<div class="log-result">&nbsp;&nbsp;{snippet}</div>')

        elif t == "asset_rendered":
            platform = e.get("platform","")
            lang     = e.get("language","")
            lines.append(f'<div class="log-image">🖼 IMAGE GENERATED → {platform} / {lang} → {Path(e["path"]).name}</div>')

        elif t == "status":
            msg  = e["message"].replace("<","&lt;")
            step = e.get("step","")
            if step == "done":
                lines.append(f'<div class="log-image">🎉 {msg}</div>')
            else:
                lines.append(f'<div class="log-status">● {msg}</div>')

        elif t == "error":
            lines.append(f'<div class="log-error">❌ {e.get("message","")[:150]}</div>')

    html = '<div class="agent-log">' + "\n".join(lines)
    html += '<div id="log-end"></div>'
    html += '</div>'
    html += '<script>var b=document.getElementById("log-end");if(b)b.scrollIntoView({behavior:"smooth"});</script>'
    return html


# ── Runner ────────────────────────────────────────────────────────────────────
def run_pipeline(brief, guidelines, ratios, ref_images, product_images):
    events  = []
    outputs = []
    done    = threading.Event()

    def on_event(e):
        events.append(e)

    ref_paths  = [f.name for f in (ref_images     or [])]
    prod_paths = [f.name for f in (product_images or [])]

    def thread_fn():
        nonlocal outputs
        try:
            agent = CampaignAgent(on_event=on_event)
            loop  = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result  = loop.run_until_complete(
                agent.run(
                    brief_text            = brief,
                    brand_guidelines      = guidelines,
                    ratio_specs           = ratios,
                    reference_image_paths = ref_paths,
                    product_image_paths   = prod_paths,
                )
            )
            outputs = result.get("outputs", [])
        except Exception as ex:
            events.append({"type": "error", "message": str(ex)})
        finally:
            done.set()

    t = threading.Thread(target=thread_fn, daemon=True)
    t.start()

    while not done.is_set():
        time.sleep(0.35)
        log = build_log_html(events[:])
        stat = f"{len(outputs)} image(s) generated · {len(events)} events"
        yield log, gr.update(interactive=False), stat

    # Final
    stat = f"✓ Complete — {len(outputs)} image(s) generated"
    yield build_log_html(events), gr.update(interactive=True), stat



def load_nike_demo():
    return NIKE_BRIEF, NIKE_GUIDELINES, NIKE_RATIOS


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(css=CSS, theme=gr.themes.Base(), title="Campaign Asset Generator") as demo:

    gr.HTML("""
    <div id="app-header">
      <h1>Campaign Asset Generator</h1>
      <span class="badge">Nano Banana Pro</span>
      <span class="badge" style="border-color:#FD79A8; color:#FD79A8;">MCP · Agentic</span>
    </div>
    """)

    with gr.Row(equal_height=False):

        # ── LEFT COLUMN: Inputs ───────────────────────────────────────────────
        with gr.Column(scale=5, min_width=380):
            gr.HTML('<span class="section-label">01 — Campaign Brief</span>')

            with gr.Tabs(elem_classes="tab-nav"):
                with gr.Tab("Brief"):
                    brief_input = gr.Textbox(
                        label="Campaign Brief",
                        placeholder="Paste your brief — brand, products, audience, tagline, concept, markets, languages…",
                        lines=13,
                        value="",
                    )
                with gr.Tab("Brand & Legal"):
                    guidelines_input = gr.Textbox(
                        label="Brand Voice + Legal Constraints",
                        placeholder="Brand voice rules, restricted language, legal copy requirements…",
                        lines=10,
                        value="",
                    )
                with gr.Tab("Ratio Specs"):
                    ratios_input = gr.Textbox(
                        label="Platform × Aspect Ratio Specs",
                        placeholder="TikTok: 9:16, 1080x1920\nInstagram: 4:5, 1080x1350\n…",
                        lines=10,
                        value="",
                    )

            gr.HTML('<span class="section-label" style="margin-top:20px; display:block;">02 — Reference Assets</span>')
            with gr.Row():
                ref_images = gr.File(
                    label="Style / Mood References",
                    file_count="multiple",
                    file_types=["image"],
                )
                product_images = gr.File(
                    label="Product Images",
                    file_count="multiple",
                    file_types=["image"],
                )

            gr.HTML("""
            <div style="background:#0E0E12; border:1px solid #242428; border-radius:8px; padding:12px 14px; margin:12px 0; font-family:'Space Mono',monospace; font-size:10px; color:#606070; line-height:1.9;">
              <span style="color:#6C5CE7;">REQUIRES</span> &nbsp;ANTHROPIC_API_KEY + GEMINI_API_KEY<br>
              Nano Banana Pro model: <span style="color:#A29BFE;">gemini-3-pro-image-preview</span><br>
              <span style="color:#FDCB6E;">→</span> Get Gemini key: aistudio.google.com/apikey
            </div>
            """)

            with gr.Row():
                demo_btn = gr.Button("← Load Nike Demo Brief", variant="secondary", size="sm")
                run_btn  = gr.Button("▶ Generate Images", variant="primary")

        # ── RIGHT COLUMN: Log + Gallery ───────────────────────────────────────
        with gr.Column(scale=7, min_width=520):
            gr.HTML('<span class="section-label">03 — Agent Reasoning</span>')
            log_html = gr.HTML(
                '<div class="agent-log"><span class="log-idle">// Agent output will stream here once you start the pipeline.</span></div>'
            )
            stat_label = gr.Markdown("", elem_id="stat-label")

    # ── Events ────────────────────────────────────────────────────────────────
    demo_btn.click(load_nike_demo, outputs=[brief_input, guidelines_input, ratios_input])

    run_btn.click(
        run_pipeline,
        inputs =[brief_input, guidelines_input, ratios_input, ref_images, product_images],
        outputs=[log_html, run_btn, stat_label],
    )


if __name__ == "__main__":
    demo.launch(
        server_port=7860,
        share=False,
        show_error=True,
    )
