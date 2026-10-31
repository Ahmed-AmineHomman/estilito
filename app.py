"""Gradio application for managing reference scenes and art styles.

This module provides a graphical interface with two main tabs:
- Reference tab: Create, edit, preview, and generate ground reference scenes.
- Art style tab: Create, edit, preview, and generate stylized scenes
  applied to one or multiple references.
Generative image tasks are powered by the Google GenAI SDK (Gemini Nano Banana models).
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
import gradio as gr

# Load environment variables from .env file at repo root
load_dotenv(Path(__file__).resolve().parent / ".env")

# File and directory paths
DEFAULT_STYLE_PATH: Path = Path("src/default_style.txt")
PROMPT_TEMPLATE_PATH: Path = Path("src/prompt_template.txt")
IMAGES_JSON_PATH: Path = Path("database/images.json")
STYLES_JSON_PATH: Path = Path("database/styles.json")
REFERENCES_DIR: Path = Path("database/references")
STYLES_DIR: Path = Path("database/styles")

DEFAULT_PROMPT_TEMPLATE: str = (
    "{concept}. Art style: {style}. {instructions}"
)

# Model and generation parameters
MODEL_CHOICES: list[tuple[str, str]] = [
    ("Nano Banana 2.1 (gemini-nano-banana-2.1)", "gemini-nano-banana-2.1"),
    ("Nano Banana Pro (gemini-3-pro-image)", "gemini-3-pro-image"),
    ("Nano Banana 2 Lite (gemini-3.1-flash-lite-image)", "gemini-3.1-flash-lite-image"),
]
DEFAULT_MODEL: str = "gemini-nano-banana-2.1"

RESOLUTION_CHOICES: list[str] = [
    "1K",
    "2K",
    "4K",
]
DEFAULT_RESOLUTION: str = "1K"

ASPECT_RATIO_CHOICES: list[str] = [
    "1:1",
    "16:9",
    "9:16",
    "4:3",
    "3:4",
    "3:2",
    "2:3",
    "21:9",
]
DEFAULT_ASPECT_RATIO: str = "16:9"


def load_default_style() -> str:
    """Load the default reference style prompt from disk.

    Returns
    -------
    str
        The content of the default style file or a fallback description.
    """
    if DEFAULT_STYLE_PATH.is_file():
        return DEFAULT_STYLE_PATH.read_text(encoding="utf-8").strip()
    return "Clean photographic storybook reference, neutral lighting, high detail."


def load_prompt_template() -> str:
    """Load the prompt template from disk.

    Returns
    -------
    str
        The content of the template file or the default template string.
    """
    if PROMPT_TEMPLATE_PATH.is_file():
        try:
            content = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8").strip()
            if content:
                return content
        except OSError:
            pass
    return DEFAULT_PROMPT_TEMPLATE


def load_json(
    path: Path,
) -> dict[str, Any]:
    """Read a JSON file and return its content as a dictionary.

    Parameters
    ----------
    path : Path
        Path to the JSON file.

    Returns
    -------
    dict of str to Any
        Parsed dictionary from the JSON file, or an empty dict on error.
    """
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as file_handle:
            data = json.load(file_handle)
            if isinstance(data, dict):
                return data
            return {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_json(
    path: Path,
    data: dict[str, Any],
) -> None:
    """Write a dictionary to a JSON file with formatting.

    Parameters
    ----------
    path : Path
        Destination path for the JSON file.
    data : dict of str to Any
        Dictionary to persist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(data, file_handle, indent=2, ensure_ascii=False)


def get_reference_choices() -> list[str]:
    """Retrieve sorted list of reference labels from JSON and directory.

    Returns
    -------
    list of str
        Sorted list of reference labels.
    """
    labels: set[str] = set()
    images = load_json(IMAGES_JSON_PATH)
    labels.update(images.keys())
    if REFERENCES_DIR.is_dir():
        for file_path in REFERENCES_DIR.glob("*.jpg"):
            labels.add(file_path.stem)
    return sorted(labels)


def get_style_choices() -> list[str]:
    """Retrieve sorted list of art style labels from JSON and directory.

    Returns
    -------
    list of str
        Sorted list of art style labels.
    """
    labels: set[str] = set()
    styles = load_json(STYLES_JSON_PATH)
    labels.update(styles.keys())
    if STYLES_DIR.is_dir():
        for file_path in STYLES_DIR.glob("**/*.jpg"):
            if file_path.stem in styles:
                labels.add(file_path.stem)
    return sorted(labels)


def clean_prompt(
    text: str,
) -> str:
    """Clean redundant labels, punctuation, and whitespace from a rendered prompt.

    Parameters
    ----------
    text : str
        Raw rendered prompt text.

    Returns
    -------
    str
        Sanitized and normalized prompt text.
    """
    text = re.sub(r"(?i)\b(?:art\s+)?style\s*:\s*(?=[.,;\n]|$)", "", text)
    text = re.sub(
        r"(?i)\b(?:global\s+)?(?:instructions?|constraints?)\s*:\s*(?=[.,;\n]|$)",
        "",
        text,
    )
    text = re.sub(
        r"(?i)\b(?:concept|scene|subject)\s*:\s*(?=[.,;\n]|$)",
        "",
        text,
    )

    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"[.,;]*\.[.,;]*", ".", text)
    text = re.sub(r",+", ",", text)
    text = re.sub(r"^[.,;:\s]+", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    lines = [line.strip(" ,;") for line in text.splitlines()]
    non_empty = [line for line in lines if line]
    result = " ".join(non_empty) if len(non_empty) <= 1 else "\n\n".join(non_empty)
    result = result.strip()
    if result and not result.endswith((".", "!", "?")):
        result += "."
    return result


def build_prompt(
    concept: str,
    style: str,
    instructions: str = "",
) -> str:
    """Build the final generation prompt using the dynamic prompt template.

    Parameters
    ----------
    concept : str
        Description of the scene subject, elements, and composition.
    style : str
        Description of the visual art style, medium, and rendering.
    instructions : str, optional
        Global constraints or generation guidelines, by default "".

    Returns
    -------
    str
        The formatted prompt for the generative model.
    """
    clean_concept = concept.strip().rstrip(".")
    clean_style = style.strip().rstrip(".")
    clean_instructions = instructions.strip().rstrip(".")

    template = load_prompt_template()
    rendered = (
        template.replace("{concept}", clean_concept)
        .replace("{style}", clean_style)
        .replace("{instructions}", clean_instructions)
        .replace("{global_instructions}", clean_instructions)
        .replace("{constraints}", clean_instructions)
    )
    return clean_prompt(rendered)


def generate_image(
    prompt: str,
    output_path: Path,
    model: str = DEFAULT_MODEL,
    resolution: str = DEFAULT_RESOLUTION,
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
) -> Path:
    """Call the Gemini API to generate an image and save it to disk.

    Parameters
    ----------
    prompt : str
        The final text prompt for image generation.
    output_path : Path
        Destination path where the generated image will be written.
    model : str, optional
        The Gemini image model ID to use, by default DEFAULT_MODEL.
    resolution : str, optional
        The image resolution ('1K', '2K', or '4K'), by default DEFAULT_RESOLUTION.
    aspect_ratio : str, optional
        The aspect ratio of the image, by default DEFAULT_ASPECT_RATIO.

    Returns
    -------
    Path
        The destination path of the generated image.

    Raises
    ------
    RuntimeError
        If the model does not return any image data.
    """
    client = genai.Client()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio,
                image_size=resolution,
            ),
        ),
    )
    if response.parts:
        for part in response.parts:
            if part.inline_data and part.inline_data.data:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(part.inline_data.data)
                return output_path

    raise RuntimeError("The model did not return any image.")


def select_reference(
    label: str | None,
) -> tuple[Any, Any, Any, str]:
    """Populate reference fields when an existing reference label is selected.

    Parameters
    ----------
    label : str or None
        The chosen reference identifier.

    Returns
    -------
    tuple of (Any, Any, Any, str)
        Tuple containing (name, prompt, image_path, status_message).
    """
    if not label:
        return "", "", None, ""

    images = load_json(IMAGES_JSON_PATH)
    if label not in images:
        return gr.update(), gr.update(), gr.update(), f"Drafting reference '{label}'."

    ref = images[label]
    ref_images = ref.get("images", [])
    image_path: str | None = None
    if isinstance(ref_images, list) and ref_images:
        first_img = ref_images[0]
        if first_img and Path(first_img).is_file():
            image_path = first_img
    elif ref.get("image") and Path(ref["image"]).is_file():
        image_path = ref["image"]
    elif (REFERENCES_DIR / f"{label}.jpg").is_file():
        image_path = str(REFERENCES_DIR / f"{label}.jpg")

    return (
        ref.get("name", ""),
        ref.get("prompt", ""),
        image_path,
        f"Loaded reference '{label}'.",
    )


def new_reference() -> tuple[Any, str, str, None, str]:
    """Reset reference fields for creating a new reference scene.

    Returns
    -------
    tuple of (Any, str, str, None, str)
        Blank fields for (ref_dropdown, name, prompt, image, status_message).
    """
    return gr.update(value=None), "", "", None, "Ready to create a new reference."


def run_reference(
    label: str | None,
    prompt: str,
    instructions: str = "",
    model: str = DEFAULT_MODEL,
    resolution: str = DEFAULT_RESOLUTION,
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
) -> tuple[str | None, str]:
    """Generate the image for a reference scene and override previous output.

    Parameters
    ----------
    label : str or None
        The reference identifier.
    prompt : str
        The scene description prompt.
    instructions : str, optional
        Global generation instructions or constraints, by default "".
    model : str, optional
        The generative model ID, by default DEFAULT_MODEL.
    resolution : str, optional
        The output image resolution, by default DEFAULT_RESOLUTION.
    aspect_ratio : str, optional
        The output image aspect ratio, by default DEFAULT_ASPECT_RATIO.

    Returns
    -------
    tuple of (str or None, str)
        The saved image path and a status message.
    """
    clean_label = (label or "").strip()
    clean_prompt = prompt.strip()
    clean_instructions = instructions.strip()
    if not clean_label:
        return None, "Error: Reference label cannot be empty."
    if not clean_prompt:
        return None, "Error: Reference prompt cannot be empty."

    default_style = load_default_style()
    full_prompt = build_prompt(
        concept=clean_prompt,
        style=default_style,
        instructions=clean_instructions,
    )
    output_path = REFERENCES_DIR / f"{clean_label}.jpg"

    try:
        generate_image(
            prompt=full_prompt,
            output_path=output_path,
            model=model,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return None, f"Generation failed: {exc}"

    return str(output_path), f"Generated reference image: '{clean_label}'."


def save_reference(
    label: str | None,
    name: str,
    prompt: str,
    image_path: str | None,
) -> tuple[Any, Any, str]:
    """Persist reference data into images.json and update dropdown choices.

    Parameters
    ----------
    label : str or None
        The reference identifier.
    name : str
        The display name.
    prompt : str
        The reference scene prompt.
    image_path : str or None
        Path to the saved image file.

    Returns
    -------
    tuple of (Any, Any, str)
        Updated ref_dropdown, updated target_ref_dropdown, and status message.
    """
    clean_label = (label or "").strip()
    if not clean_label:
        return gr.update(), gr.update(), "Error: Label is required to save."

    (STYLES_DIR / clean_label).mkdir(parents=True, exist_ok=True)

    images = load_json(IMAGES_JSON_PATH)
    images[clean_label] = {
        "label": clean_label,
        "name": name.strip(),
        "prompt": prompt.strip(),
        "image": image_path or "",
    }
    save_json(IMAGES_JSON_PATH, images)
    all_labels = get_reference_choices()

    return (
        gr.update(choices=all_labels, value=clean_label),
        gr.update(choices=all_labels),
        f"Reference '{clean_label}' saved successfully.",
    )


def select_style(
    style_label: str | None,
    target_ref_label: str | None,
) -> tuple[Any, Any, Any, str]:
    """Populate art style fields when an existing style label is selected.

    Parameters
    ----------
    style_label : str or None
        The chosen style identifier.
    target_ref_label : str or None
        The currently selected target reference label, if any.

    Returns
    -------
    tuple of (Any, Any, Any, str)
        Tuple containing (name, prompt, image_path, status_message).
    """
    if not style_label:
        return "", "", None, ""

    styles = load_json(STYLES_JSON_PATH)
    if style_label not in styles:
        return gr.update(), gr.update(), gr.update(), f"Drafting style '{style_label}'."

    style = styles[style_label]
    style_images = style.get("images", {})

    image_path: str | None = None
    if isinstance(style_images, dict) and target_ref_label:
        candidate = style_images.get(target_ref_label)
        if candidate and Path(candidate).is_file():
            image_path = candidate

    return (
        style.get("name", ""),
        style.get("prompt", ""),
        image_path,
        f"Loaded style '{style_label}'.",
    )


def change_target_reference(
    style_label: str | None,
    target_ref_label: str | None,
) -> str | None:
    """Update preview image when switching target reference in the style tab.

    Parameters
    ----------
    style_label : str or None
        The active style identifier.
    target_ref_label : str or None
        The newly selected target reference identifier.

    Returns
    -------
    str or None
        Path to the corresponding styled image, or None if not found.
    """
    if not style_label or not target_ref_label:
        return None

    styles = load_json(STYLES_JSON_PATH)
    style = styles.get(style_label, {})
    style_images = style.get("images", {})
    if isinstance(style_images, dict):
        candidate = style_images.get(target_ref_label)
        if candidate and Path(candidate).is_file():
            return candidate

    return None


def new_style() -> tuple[Any, str, str, None, str]:
    """Reset style fields for creating a new art style.

    Returns
    -------
    tuple of (Any, str, str, None, str)
        Blank fields for (style_dropdown, name, prompt, image, status_message).
    """
    return gr.update(value=None), "", "", None, "Ready to create a new art style."


def run_style(
    style_label: str | None,
    style_prompt: str,
    target_ref_label: str | None,
    instructions: str = "",
    model: str = DEFAULT_MODEL,
    resolution: str = DEFAULT_RESOLUTION,
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
) -> tuple[str | None, str]:
    """Generate a stylized image for a reference scene, overriding previous image.

    Parameters
    ----------
    style_label : str or None
        The art style identifier.
    style_prompt : str
        The art style prompt.
    target_ref_label : str or None
        The target reference identifier to be styled.
    instructions : str, optional
        Global generation instructions or constraints, by default "".
    model : str, optional
        The generative model ID, by default DEFAULT_MODEL.
    resolution : str, optional
        The output image resolution, by default DEFAULT_RESOLUTION.
    aspect_ratio : str, optional
        The output image aspect ratio, by default DEFAULT_ASPECT_RATIO.

    Returns
    -------
    tuple of (str or None, str)
        The saved styled image path and a status message.
    """
    clean_style_label = (style_label or "").strip()
    clean_style_prompt = style_prompt.strip()
    clean_target_ref = (target_ref_label or "").strip()
    clean_instructions = instructions.strip()
    if not clean_style_label:
        return None, "Error: Art style label cannot be empty."
    if not clean_style_prompt:
        return None, "Error: Art style prompt cannot be empty."
    if not clean_target_ref:
        return None, "Error: Please select a target reference first."

    images = load_json(IMAGES_JSON_PATH)
    ref = images.get(clean_target_ref)
    if not ref:
        return None, f"Error: Target reference '{clean_target_ref}' not found."

    ref_prompt = ref.get("prompt", "")
    full_prompt = build_prompt(
        concept=ref_prompt,
        style=clean_style_prompt,
        instructions=clean_instructions,
    )
    output_path = STYLES_DIR / clean_target_ref / f"{clean_style_label}.jpg"

    try:
        generate_image(
            prompt=full_prompt,
            output_path=output_path,
            model=model,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return None, f"Generation failed: {exc}"

    styles = load_json(STYLES_JSON_PATH)
    if clean_style_label in styles:
        current_images = styles[clean_style_label].get("images", {})
        if not isinstance(current_images, dict):
            current_images = {}
        current_images[clean_target_ref] = str(output_path)
        styles[clean_style_label]["images"] = current_images
        save_json(STYLES_JSON_PATH, styles)

    return (
        str(output_path),
        f"Generated style '{clean_style_label}' for reference '{clean_target_ref}'.",
    )


def save_style(
    style_label: str | None,
    name: str,
    prompt: str,
) -> tuple[Any, str]:
    """Persist art style data into styles.json and update dropdown choices.

    Parameters
    ----------
    style_label : str or None
        The art style identifier.
    name : str
        The display name.
    prompt : str
        The art style prompt.

    Returns
    -------
    tuple of (Any, str)
        Updated style_dropdown and status message.
    """
    clean_label = (style_label or "").strip()
    if not clean_label:
        return gr.update(), "Error: Style label is required to save."

    styles = load_json(STYLES_JSON_PATH)
    existing_images = styles.get(clean_label, {}).get("images", {})
    if not isinstance(existing_images, dict):
        existing_images = {}
    styles[clean_label] = {
        "label": clean_label,
        "name": name.strip(),
        "prompt": prompt.strip(),
        "images": existing_images,
    }
    save_json(STYLES_JSON_PATH, styles)
    all_styles = get_style_choices()

    return (
        gr.update(choices=all_styles, value=clean_label),
        f"Art style '{clean_label}' saved successfully.",
    )


def build_app() -> gr.Blocks:
    """Build and return the Gradio application layout.

    Returns
    -------
    gr.Blocks
        The instantiated Gradio interface.
    """
    REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
    STYLES_DIR.mkdir(parents=True, exist_ok=True)

    initial_images = load_json(IMAGES_JSON_PATH)
    initial_styles = load_json(STYLES_JSON_PATH)

    ref_choices = get_reference_choices()
    style_choices = get_style_choices()

    for ref_label in ref_choices:
        (STYLES_DIR / ref_label).mkdir(parents=True, exist_ok=True)

    default_ref_key = ref_choices[0] if ref_choices else None
    default_ref_data = (
        initial_images.get(default_ref_key, {}) if default_ref_key else {}
    )
    default_ref_images = default_ref_data.get("images", [])
    default_ref_image = (
        default_ref_images[0]
        if (isinstance(default_ref_images, list) and default_ref_images)
        else default_ref_data.get("image")
    )
    if not default_ref_image or not Path(default_ref_image).is_file():
        default_ref_image = None

    default_style_key = style_choices[0] if style_choices else None
    default_style_data = (
        initial_styles.get(default_style_key, {}) if default_style_key else {}
    )
    default_style_images = default_style_data.get("images", {})
    default_style_image = (
        default_style_images.get(default_ref_key)
        if isinstance(default_style_images, dict) and default_ref_key
        else None
    )
    if not default_style_image or not Path(default_style_image).is_file():
        default_style_image = None

    with gr.Blocks(title="Art Style Studio") as demo:
        gr.Markdown("# Art Style Studio")

        with gr.Accordion("Generation Parameters", open=False):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    global_instructions_input = gr.Textbox(
                        label="Global Instructions",
                        placeholder="e.g. borderless image, no frame, uncluttered scene",
                        lines=7,
                    )
                with gr.Column(scale=1):
                    model_dropdown = gr.Dropdown(
                        label="Model",
                        choices=MODEL_CHOICES,
                        value=DEFAULT_MODEL,
                    )
                    resolution_dropdown = gr.Dropdown(
                        label="Resolution",
                        choices=RESOLUTION_CHOICES,
                        value=DEFAULT_RESOLUTION,
                    )
                    aspect_ratio_dropdown = gr.Dropdown(
                        label="Aspect Ratio",
                        choices=ASPECT_RATIO_CHOICES,
                        value=DEFAULT_ASPECT_RATIO,
                    )

        with gr.Tabs():
            # Tab 1: References
            with gr.Tab("References"):
                with gr.Row():
                    ref_dropdown = gr.Dropdown(
                        label="Reference",
                        choices=ref_choices,
                        value=default_ref_key,
                        allow_custom_value=True,
                        scale=6,
                    )
                    with gr.Column(scale=1):
                        new_ref_btn = gr.Button("New")
                        save_ref_btn = gr.Button("Save", variant="secondary")

                with gr.Row():
                    with gr.Column(scale=1):
                        ref_name_input = gr.Textbox(
                            label="Name",
                            value=default_ref_data.get("name", ""),
                            placeholder="e.g. Cozy Autumn Cottage",
                        )
                        ref_prompt_input = gr.Textbox(
                            label="Prompt",
                            lines=8,
                            value=default_ref_data.get("prompt", ""),
                            placeholder="Describe the reference scene...",
                        )
                    with gr.Column(scale=2):
                        ref_image_preview = gr.Image(
                            label="Image",
                            type="filepath",
                            value=default_ref_image,
                        )
                        run_ref_btn = gr.Button("Generate", variant="primary")
                        ref_status = gr.Markdown()

            # Tab 2: Art Styles
            with gr.Tab("Styles"):
                with gr.Row():
                    style_dropdown = gr.Dropdown(
                        label="Style",
                        choices=style_choices,
                        value=default_style_key,
                        allow_custom_value=True,
                        scale=3,
                    )
                    target_ref_dropdown = gr.Dropdown(
                        label="Reference",
                        choices=ref_choices,
                        value=default_ref_key,
                        scale=3,
                    )
                    with gr.Column(scale=1):
                        new_style_btn = gr.Button("New")
                        save_style_btn = gr.Button("Save", variant="secondary")

                with gr.Row():
                    with gr.Column(scale=1):
                        style_name_input = gr.Textbox(
                            label="Name",
                            value=default_style_data.get("name", ""),
                            placeholder="e.g. Whimsical 3D Concept Art",
                        )
                        style_prompt_input = gr.Textbox(
                            label="Prompt",
                            lines=8,
                            value=default_style_data.get("prompt", ""),
                            placeholder="Describe the art style...",
                        )
                    with gr.Column(scale=2):
                        style_image_preview = gr.Image(
                            label="Image",
                            type="filepath",
                            value=default_style_image,
                        )
                        run_style_btn = gr.Button("Generate", variant="primary")
                        style_status = gr.Markdown()

        # Reference Tab Events
        ref_dropdown.change(
            fn=select_reference,
            inputs=[ref_dropdown],
            outputs=[
                ref_name_input,
                ref_prompt_input,
                ref_image_preview,
                ref_status,
            ],
        )

        new_ref_btn.click(
            fn=new_reference,
            inputs=[],
            outputs=[
                ref_dropdown,
                ref_name_input,
                ref_prompt_input,
                ref_image_preview,
                ref_status,
            ],
        )

        run_ref_btn.click(
            fn=run_reference,
            inputs=[
                ref_dropdown,
                ref_prompt_input,
                global_instructions_input,
                model_dropdown,
                resolution_dropdown,
                aspect_ratio_dropdown,
            ],
            outputs=[ref_image_preview, ref_status],
        )

        save_ref_btn.click(
            fn=save_reference,
            inputs=[
                ref_dropdown,
                ref_name_input,
                ref_prompt_input,
                ref_image_preview,
            ],
            outputs=[ref_dropdown, target_ref_dropdown, ref_status],
        )

        # Art Style Tab Events
        style_dropdown.change(
            fn=select_style,
            inputs=[style_dropdown, target_ref_dropdown],
            outputs=[
                style_name_input,
                style_prompt_input,
                style_image_preview,
                style_status,
            ],
        )

        target_ref_dropdown.change(
            fn=change_target_reference,
            inputs=[style_dropdown, target_ref_dropdown],
            outputs=[style_image_preview],
        )

        new_style_btn.click(
            fn=new_style,
            inputs=[],
            outputs=[
                style_dropdown,
                style_name_input,
                style_prompt_input,
                style_image_preview,
                style_status,
            ],
        )

        run_style_btn.click(
            fn=run_style,
            inputs=[
                style_dropdown,
                style_prompt_input,
                target_ref_dropdown,
                global_instructions_input,
                model_dropdown,
                resolution_dropdown,
                aspect_ratio_dropdown,
            ],
            outputs=[style_image_preview, style_status],
        )

        save_style_btn.click(
            fn=save_style,
            inputs=[
                style_dropdown,
                style_name_input,
                style_prompt_input,
            ],
            outputs=[style_dropdown, style_status],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
