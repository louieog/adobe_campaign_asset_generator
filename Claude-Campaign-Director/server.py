"""
Campaign Asset Generator — MCP Server v2
Tools: parse_brief, analyze_references, generate_copy, score_copy, build_image_prompt, generate_image
Image generation powered by Nano Banana Pro (gemini-3-pro-image-preview)
"""

import asyncio
import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import anthropic
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# ── Init ──────────────────────────────────────────────────────────────────────
app = Server("campaign-asset-generator")
claude = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
EXPORTS_DIR = Path(__file__).parent.parent / "exports"
EXPORTS_DIR.mkdir(exist_ok=True)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL_PRIMARY  = os.environ.get("GEMINI_MODEL", "gemini-3-pro-image-preview")
GEMINI_MODEL_FALLBACK = "gemini-2.5-flash-image"

# Nano Banana Pro natively supports these aspect ratios
RATIO_MAP = {
    "9:16": "9:16",
    "4:5":  "4:5",
    "1:1":  "1:1",
    "16:9": "16:9",
    "3:4":  "3:4",
}


# ── Tool Definitions ───────────────────────────────────────────────────────────
@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="parse_brief",
            description="Parse a raw campaign brief into structured JSON with brand info, audience, platforms, languages, aspect ratios, and guidelines.",
            inputSchema={
                "type": "object",
                "properties": {
                    "brief_text":      {"type": "string", "description": "Raw campaign brief text"},
                    "brand_guidelines":{"type": "string", "description": "Brand and legal guidelines text"},
                    "ratio_specs":     {"type": "string", "description": "Aspect ratio specifications per platform"},
                },
                "required": ["brief_text"],
            },
        ),
        Tool(
            name="analyze_references",
            description=(
                "Analyze reference images (product shots, brand assets, mood boards) to extract "
                "visual style, color palette, composition, lighting, and mood. "
                "Returns a rich style descriptor used to inform Nano Banana Pro image generation prompts."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "image_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Absolute file paths to reference/product images (up to 6)",
                    },
                    "brand":    {"type": "string", "description": "Brand name for context"},
                    "campaign": {"type": "string", "description": "Campaign name/concept for context"},
                },
                "required": ["image_paths"],
            },
        ),
        Tool(
            name="generate_copy",
            description="Generate platform-optimized ad copy for a specific platform and language. Returns headline, subheadline, body, and CTA with character counts.",
            inputSchema={
                "type": "object",
                "properties": {
                    "brief_json":       {"type": "string", "description": "Structured brief JSON from parse_brief"},
                    "platform":         {"type": "string", "description": "Target platform (e.g. TikTok, Instagram, YouTube Shorts, X (Twitter), LinkedIn)"},
                    "language":         {"type": "string", "description": "Language code (EN, ES, FR, JA, ZH, PT, DE)"},
                    "brand_guidelines": {"type": "string", "description": "Brand voice and legal constraints"},
                },
                "required": ["brief_json", "platform", "language"],
            },
        ),
        Tool(
            name="score_copy",
            description=(
                "Score generated copy 0-100 across 5 dimensions. "
                "Returns pass/fail (threshold 75) and exact revision text. "
                "ALWAYS call after generate_copy. If score < 75, apply revisions and re-score."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "copy_json":        {"type": "string", "description": "Copy JSON from generate_copy"},
                    "platform":         {"type": "string"},
                    "language":         {"type": "string"},
                    "brand_guidelines": {"type": "string"},
                    "brief_json":       {"type": "string"},
                },
                "required": ["copy_json", "platform", "language"],
            },
        ),
        Tool(
            name="build_image_prompt",
            description=(
                "Craft a detailed, production-quality image generation prompt for Nano Banana Pro. "
                "Synthesizes approved copy, style descriptor from reference analysis, platform format, "
                "and brand identity into a single richly-detailed prompt with text-in-image instructions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "copy_json":        {"type": "string", "description": "Approved copy JSON"},
                    "brief_json":       {"type": "string", "description": "Structured campaign brief JSON"},
                    "style_descriptor": {"type": "string", "description": "Visual style JSON from analyze_references"},
                    "platform":         {"type": "string", "description": "Target platform"},
                    "ratio":            {"type": "string", "description": "Aspect ratio e.g. 9:16"},
                    "language":         {"type": "string", "description": "Language code e.g. EN"},
                    "brand_guidelines": {"type": "string", "description": "Brand and legal guidelines"},
                },
                "required": ["copy_json", "brief_json", "platform", "ratio"],
            },
        ),
        Tool(
            name="generate_image",
            description=(
                "Generate a complete campaign ad image using Nano Banana Pro (gemini-3-pro-image-preview). "
                "Optionally accepts reference/product images as visual context. "
                "Saves PNG to exports directory and returns the file path."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt":       {"type": "string", "description": "Full image generation prompt from build_image_prompt"},
                    "aspect_ratio": {"type": "string", "description": "e.g. 9:16, 4:5, 1:1, 16:9"},
                    "platform":     {"type": "string"},
                    "language":     {"type": "string"},
                    "brand":        {"type": "string"},
                    "reference_image_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: product/reference image paths for visual context (style anchoring)",
                    },
                    "resolution":   {"type": "string", "description": "Resolution hint e.g. 1080x1920 (filename only)"},
                },
                "required": ["prompt", "aspect_ratio", "platform", "brand"],
            },
        ),
    ]


# ── Dispatcher ────────────────────────────────────────────────────────────────
@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    dispatch = {
        "parse_brief":        _parse_brief,
        "analyze_references": _analyze_references,
        "generate_copy":      _generate_copy,
        "score_copy":         _score_copy,
        "build_image_prompt": _build_image_prompt,
        "generate_image":     _generate_image,
    }
    fn = dispatch.get(name)
    if not fn:
        return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]
    return await fn(**arguments)


# ── parse_brief ───────────────────────────────────────────────────────────────
async def _parse_brief(brief_text: str, brand_guidelines: str = "", ratio_specs: str = "") -> list[TextContent]:
    prompt = f"""Parse this campaign brief into a structured JSON object. Extract ALL details.

BRIEF:
{brief_text}

BRAND GUIDELINES:
{brand_guidelines or "Not provided"}

RATIO SPECS:
{ratio_specs or "Not provided"}

Return ONLY valid JSON:
{{
  "brand": "string",
  "campaign_name": "string",
  "campaign_period": "string",
  "tagline": "string",
  "concept": "string",
  "products": ["list of product names"],
  "target_audience": {{
    "primary": "description",
    "secondary": "description",
    "age_range": "e.g. 18-35",
    "psychographic": "brief descriptor"
  }},
  "markets": ["list of cities/regions"],
  "languages": ["EN", "ES", ...],
  "platforms": [
    {{
      "name": "TikTok",
      "format": "Full-Screen Video",
      "ratio": "9:16",
      "resolution": "1080x1920"
    }}
  ],
  "brand_voice": "brief descriptor",
  "visual_identity": "any visual cues from brief (colors, aesthetic, photography style)",
  "legal_constraints": ["list of constraints"],
  "color_palette_hints": ["any color mentions"]
}}"""

    resp = claude.messages.create(
        model="claude-opus-4-5",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    return [TextContent(type="text", text=_strip_fences(resp.content[0].text))]


# ── analyze_references ────────────────────────────────────────────────────────
async def _analyze_references(image_paths: list[str], brand: str = "", campaign: str = "") -> list[TextContent]:
    content = []
    loaded = []

    for path in image_paths[:6]:
        try:
            with open(path, "rb") as f:
                data = base64.standard_b64encode(f.read()).decode()
            ext = Path(path).suffix.lower().lstrip(".")
            media_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}
            media_type = media_map.get(ext, "image/jpeg")
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": data}
            })
            loaded.append(path)
        except Exception:
            pass

    if not content:
        return [TextContent(type="text", text=json.dumps({
            "style_descriptor": "Clean, modern performance photography. Dramatic lighting, high contrast. Minimal backgrounds.",
            "color_palette": ["#0A0A0A", "#FFFFFF", "#FF3B00", "#F5F5F0"],
            "dominant_color": "#0A0A0A",
            "mood": "energetic, aspirational, precision-engineered",
            "composition": "Hero product positioned dynamically. Copy zone lower-left or lower third.",
            "lighting": "Hard directional studio light. Cinematic contrast. Deep shadows.",
            "background_treatment": "Near-black or clean white with subtle gradient.",
            "photography_style": "Photorealistic editorial advertising photography",
            "key_visual_elements": ["motion", "speed", "close crop", "dramatic shadow", "product hero"],
            "typography_treatment": "Ultra-bold condensed sans-serif headline. Clean body text.",
            "image_gen_prompt_fragment": "dramatic studio lighting, hard shadows, high contrast, cinematic color grade, photorealistic, shot on Phase One IQ4, editorial advertising photography, sharp focus, 4K",
            "product_integration_notes": "Product as hero — well-lit, isolated against complementary background. Dynamic angle.",
            "notes": "No references provided — performance brand defaults applied."
        }))]

    content.append({
        "type": "text",
        "text": f"""You are a senior art director analyzing {len(loaded)} reference image(s) for {brand or 'this brand'} — {campaign or 'a campaign'}.

These images will directly inform AI image generation prompts for a professional ad campaign.
Analyze with the precision of someone briefing a photographer and CGI team.

Return ONLY valid JSON:
{{
  "style_descriptor": "3-4 sentence rich description. Be specific: lighting quality, color temperature, texture, energy, photographic approach, what makes this brand's look distinctive.",
  "color_palette": ["#hex1", "#hex2", "#hex3", "#hex4", "#hex5"],
  "dominant_color": "#hex",
  "mood": "2-4 descriptive words",
  "composition": "Describe layout, framing, product placement, negative space, camera angle, depth of field",
  "lighting": "Lighting style — direction, hard/soft, color temperature, shadow behavior, any practical lights visible",
  "background_treatment": "Color, texture, gradient, environmental context, how product separates from background",
  "photography_style": "Photorealistic vs stylized? Studio vs environmental? Action vs static? Film stock feel?",
  "key_visual_elements": ["5-8 specific recurring visual elements that define this brand's visual language"],
  "typography_treatment": "Font weight, case, placement, color, relationship to imagery if text visible in refs",
  "image_gen_prompt_fragment": "A dense, comma-separated visual descriptor list (40-60 words) ready to append to any image gen prompt. Include: lighting terms, photography style, camera references, color treatment, mood words, quality terms like 'photorealistic 4K commercial photography'",
  "product_integration_notes": "How to show the product in the ad: hero object, in-use, isolated, with athlete, environmental"
}}"""
    })

    resp = claude.messages.create(
        model="claude-opus-4-5",
        max_tokens=1500,
        messages=[{"role": "user", "content": content}],
    )
    return [TextContent(type="text", text=_strip_fences(resp.content[0].text))]


# ── generate_copy ─────────────────────────────────────────────────────────────
async def _generate_copy(brief_json: str, platform: str, language: str, brand_guidelines: str = "") -> list[TextContent]:
    CHAR_LIMITS = {
        "TikTok":         {"headline": 40,  "subheadline": 80,  "body": 100, "cta": 20},
        "Instagram":      {"headline": 50,  "subheadline": 100, "body": 140, "cta": 25},
        "YouTube Shorts": {"headline": 45,  "subheadline": 90,  "body": 120, "cta": 20},
        "X (Twitter)":    {"headline": 50,  "subheadline": 80,  "body": 100, "cta": 20},
        "LinkedIn":       {"headline": 60,  "subheadline": 120, "body": 170, "cta": 30},
    }
    lim = CHAR_LIMITS.get(platform, CHAR_LIMITS["Instagram"])

    prompt = f"""You are a senior creative copywriter at a world-class ad agency. Write native advertising copy — not translations.

CAMPAIGN BRIEF:
{brief_json}

BRAND & LEGAL GUIDELINES:
{brand_guidelines or "Infer voice from brief."}

PLATFORM: {platform}
LANGUAGE: {language}
LIMITS: Headline ≤{lim['headline']} chars, Subheadline ≤{lim['subheadline']} chars, Body ≤{lim['body']} chars, CTA ≤{lim['cta']} chars

RULES:
1. Write entirely in {language} — culturally adapted, not translated
2. Headline must hit hard at large type size. Prefer ≤5 power words.
3. Honor the campaign tagline verbatim or note cultural adaptation
4. CTA must start with an action verb. Be urgent.
5. No passive voice. Active, direct, athletic energy.
6. Comply with all legal constraints from guidelines.

Return ONLY valid JSON:
{{
  "platform": "{platform}",
  "language": "{language}",
  "headline": "string",
  "subheadline": "string",
  "body": "string",
  "cta": "string",
  "char_counts": {{
    "headline": 0,
    "subheadline": 0,
    "body": 0,
    "cta": 0
  }},
  "tagline_treatment": "how the campaign tagline was handled",
  "cultural_notes": "adaptation choices made"
}}"""

    resp = claude.messages.create(
        model="claude-opus-4-5",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    return [TextContent(type="text", text=_strip_fences(resp.content[0].text))]


# ── score_copy ────────────────────────────────────────────────────────────────
async def _score_copy(copy_json: str, platform: str, language: str, brand_guidelines: str = "", brief_json: str = "") -> list[TextContent]:
    prompt = f"""You are a creative director and brand compliance officer reviewing ad copy before production.

BRIEF:
{brief_json or "Not provided"}

GUIDELINES:
{brand_guidelines or "Not provided"}

COPY UNDER REVIEW:
{copy_json}

Platform: {platform} | Language: {language}

Score on 5 dimensions (0–20 each = 100 total):
1. Brand Voice Match — sounds authentically like the brand
2. Platform Fit — length, tone, energy appropriate for {platform}
3. Cultural Authenticity — feels natural in {language}, not translated
4. Campaign Alignment — reflects brief concept, tagline, products
5. Conversion Potential — compelling, specific, drives action

Passing score: 75+. If below 75, provide EXACT replacement text (not suggestions).

Return ONLY valid JSON:
{{
  "total_score": 0,
  "scores": {{
    "brand_voice": 0,
    "platform_fit": 0,
    "cultural_authenticity": 0,
    "campaign_alignment": 0,
    "conversion_potential": 0
  }},
  "pass": true,
  "issues": ["specific issue 1", "specific issue 2"],
  "revised_copy": {{
    "headline": "exact revised text or null",
    "subheadline": "exact revised text or null",
    "body": "exact revised text or null",
    "cta": "exact revised text or null"
  }},
  "approval_notes": "one-sentence creative director verdict"
}}"""

    resp = claude.messages.create(
        model="claude-opus-4-5",
        max_tokens=900,
        messages=[{"role": "user", "content": prompt}],
    )
    return [TextContent(type="text", text=_strip_fences(resp.content[0].text))]


# ── build_image_prompt ────────────────────────────────────────────────────────
async def _build_image_prompt(
    copy_json: str,
    brief_json: str,
    platform: str,
    ratio: str,
    style_descriptor: str = "",
    language: str = "EN",
    brand_guidelines: str = "",
) -> list[TextContent]:

    prompt = f"""You are a senior art director crafting a prompt for Nano Banana Pro (gemini-3-pro-image-preview) — Google's professional image generation model with advanced reasoning and high-fidelity text rendering.

Your prompt will produce a COMPLETE, publication-ready social media ad. Not a background. Not a product shot. A finished ad.

CAMPAIGN BRIEF:
{brief_json}

APPROVED COPY (must appear verbatim in the image):
{copy_json}

VISUAL STYLE (from reference image analysis):
{style_descriptor or "Use brand-appropriate defaults: cinematic, high-contrast, aspirational."}

BRAND GUIDELINES:
{brand_guidelines or "Infer from brief."}

PLATFORM: {platform}
ASPECT RATIO: {ratio}
LANGUAGE: {language}

Craft a single richly detailed image generation prompt. Structure it as a natural paragraph, not a list.

The prompt MUST:
1. Instruct the model to render ALL copy text directly in the image (Nano Banana Pro handles text well)
2. Specify exact text content: what the headline says, subheadline, body, CTA — quoted exactly from the copy JSON
3. Describe text styling: font weight (ultra-bold/condensed for headline), case, color, placement in the composition
4. Define the visual scene: atmosphere, environment, product, human element if any
5. Reference the visual style analysis: lighting, color palette, photography approach
6. Specify composition for the ratio: e.g. for 9:16 "lower-third text zone, product hero upper two-thirds"
7. Close with Nano Banana Pro quality triggers: "gemini image quality, photorealistic 4K commercial advertising photography, razor sharp focus, professional color grading"

Also include a negative prompt to keep quality high.

Return ONLY valid JSON:
{{
  "prompt": "The complete image generation prompt — 180-280 words, very specific",
  "aspect_ratio": "{ratio}",
  "negative_prompt": "blurry text, warped letters, misspelled words, watermark, low quality, amateur, stock photo, generic, overexposed, flat lighting, cartoon unless specified",
  "art_direction_notes": "2-3 sentences on the key creative choices and why"
}}"""

    resp = claude.messages.create(
        model="claude-opus-4-5",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    return [TextContent(type="text", text=_strip_fences(resp.content[0].text))]


# ── generate_image ────────────────────────────────────────────────────────────
async def _generate_image(
    prompt: str,
    aspect_ratio: str,
    platform: str,
    brand: str,
    language: str = "EN",
    reference_image_paths: list[str] = None,
    resolution: str = "",
) -> list[TextContent]:
    reference_image_paths = reference_image_paths or []

    if not GEMINI_API_KEY:
        return [TextContent(type="text", text=json.dumps({
            "success": False,
            "error": "GEMINI_API_KEY not set.",
            "fix": "export GEMINI_API_KEY=AIza... (get from https://aistudio.google.com/apikey)"
        }))]

    try:
        from google import genai
        from google.genai import types as gtypes
    except ImportError:
        return [TextContent(type="text", text=json.dumps({
            "success": False,
            "error": "google-genai not installed",
            "fix": "pip install google-genai"
        }))]

    # Build multipart content
    contents_parts = []

    # Prepend reference images for style anchoring
    for img_path in reference_image_paths[:4]:
        try:
            with open(img_path, "rb") as f:
                img_bytes = f.read()
            ext = Path(img_path).suffix.lower().lstrip(".")
            mt = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")
            contents_parts.append(gtypes.Part.from_bytes(data=img_bytes, mime_type=mt))
        except Exception:
            pass

    # Text prompt
    contents_parts.append(gtypes.Part.from_text(text=prompt))

    # Output filename
    ts = int(time.time())
    safe_p = re.sub(r"[^a-z0-9]", "-", platform.lower())
    safe_r = aspect_ratio.replace(":", "")
    filename = f"{brand.lower()}_{safe_p}_{language.upper()}_{safe_r}_{ts}.png"
    output_path = EXPORTS_DIR / filename

    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    models_to_try = [GEMINI_MODEL_PRIMARY, GEMINI_MODEL_FALLBACK]
    last_error = ""

    for model in models_to_try:
        try:
            response = gemini_client.models.generate_content(
                model=model,
                contents=contents_parts,
                config=gtypes.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                    image_config=gtypes.ImageConfig(
                        aspect_ratio=RATIO_MAP.get(aspect_ratio, "1:1"),
                    ),
                ),
            )

            saved = _extract_and_save_image(response, output_path)

            if saved:
                return [TextContent(type="text", text=json.dumps({
                    "success": True,
                    "output_path": str(output_path),
                    "filename": filename,
                    "platform": platform,
                    "language": language,
                    "aspect_ratio": aspect_ratio,
                    "model": model,
                }))]
            else:
                last_error = f"{model}: No image data in response"

        except Exception as e:
            last_error = f"{model}: {e}"

    # Both models failed — return the prompt so the agent log can display it
    return [TextContent(type="text", text=json.dumps({
        "success": False,
        "error": last_error,
        "prompt_for_review": prompt,
        "platform": platform,
        "language": language,
        "aspect_ratio": aspect_ratio,
    }))]


def _extract_and_save_image(response, output_path: Path) -> bool:
    """Try multiple response shapes to extract image bytes and save."""
    # Shape 1: response.candidates[0].content.parts
    try:
        for part in response.candidates[0].content.parts:
            if hasattr(part, "inline_data") and part.inline_data:
                raw = part.inline_data.data
                if isinstance(raw, str):
                    raw = base64.b64decode(raw)
                output_path.write_bytes(raw)
                return True
    except Exception:
        pass

    # Shape 2: response.parts (older SDK)
    try:
        for part in response.parts:
            if hasattr(part, "inline_data") and part.inline_data:
                raw = part.inline_data.data
                if isinstance(raw, str):
                    raw = base64.b64decode(raw)
                output_path.write_bytes(raw)
                return True
    except Exception:
        pass

    # Shape 3: part.as_image() → PIL
    try:
        from PIL import Image as PILImage
        for part in response.candidates[0].content.parts:
            if hasattr(part, "as_image"):
                img = part.as_image()
                img.save(str(output_path), "PNG")
                return True
    except Exception:
        pass

    return False


# ── Helpers ───────────────────────────────────────────────────────────────────
def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


# ── Entry ─────────────────────────────────────────────────────────────────────
async def main():
    async with stdio_server() as (r, w):
        await app.run(r, w, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
