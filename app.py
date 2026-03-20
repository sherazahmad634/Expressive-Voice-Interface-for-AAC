import os
import sys
import time
import tempfile
import traceback
import gc
import pathlib

# Must be set before transformers/chatterbox imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def patch_perth_pkg_resources():
    """
    Patch perth's pkg_resources import in a way that works on Spaces.
    This replaces the Colab hardcoded Python 3.12 path patch.
    """
    original = "from pkg_resources import resource_filename"
    patched = (
        "try:\n"
        "    from pkg_resources import resource_filename\n"
        "except Exception:\n"
        "    import importlib.resources as _ir\n"
        "    def resource_filename(pkg, path):\n"
        "        return str(_ir.files(pkg).joinpath(path))\n"
    )

    for p in sys.path:
        if not p:
            continue
        target = pathlib.Path(p) / "perth" / "perth_net" / "__init__.py"
        if target.exists():
            text = target.read_text(encoding="utf-8")
            if original in text and "importlib.resources as _ir" not in text:
                target.write_text(text.replace(original, patched), encoding="utf-8")
                print("✅ perth patched:", target)
            else:
                print("ℹ️ perth already patched:", target)
            return

    print("ℹ️ perth file not found; skipping patch")


patch_perth_pkg_resources()

import torch
import torchaudio as ta
import gradio as gr
from huggingface_hub import snapshot_download
from chatterbox.tts_turbo import ChatterboxTurboTTS


# -----------------------------
# Model setup
# -----------------------------
MODEL_ID = "ResembleAI/chatterbox-turbo"
CACHE_DIR = os.path.join(os.getcwd(), "hf_cache")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = None
local_path = None


def get_model():
    """
    Lazy-load the model on first use.
    This is better on Spaces than downloading/loading at import time.
    """
    global model, local_path

    if model is None:
        print(f"Downloading {MODEL_ID} ...")
        local_path = snapshot_download(
            repo_id=MODEL_ID,
            token=False,
            cache_dir=CACHE_DIR,
            local_files_only=False,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"],
        )
        print("✅ Snapshot path:", local_path)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        print(f"Loading model on: {DEVICE}")
        model = ChatterboxTurboTTS.from_local(local_path, device=DEVICE)
        print(f"✅ Model loaded | sample rate: {model.sr}")

    return model


# -----------------------------
# Helper functions
# -----------------------------
def first_path(value):
    """
    Robustly extract a filepath from a Gradio File value.
    Handles:
    - single string path
    - list/tuple of paths
    - file-like wrappers with .path or .name
    """
    if value is None:
        return None

    if isinstance(value, str):
        return value

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        first = value[0]
        if isinstance(first, str):
            return first
        if hasattr(first, "path"):
            return first.path
        if hasattr(first, "name"):
            return first.name
        return None

    if hasattr(value, "path"):
        return value.path

    if hasattr(value, "name"):
        return value.name

    return None


def validate_ref_audio(ref_audio_path):
    """Returns an error string, or None if the audio is valid."""
    if not ref_audio_path:
        return None
    try:
        info = ta.info(ref_audio_path)
        dur = info.num_frames / float(info.sample_rate)
        if dur < 5.0:
            return f"Reference audio too short ({dur:.2f}s). Please upload a 5–10s clip."
        return None
    except Exception as e:
        return f"Could not read reference audio. Try a WAV file. Details: {e}"


def make_unique_wav_path():
    return os.path.join(
        tempfile.gettempdir(),
        f"chatterbox_{os.getpid()}_{int(time.time() * 1000)}.wav"
    )


def generate_speech(text, ref_audio_path, exaggeration, cfg_weight):
    t0 = time.time()
    try:
        text = (text or "").strip()
        if not text:
            yield None, "⚠️ No text provided."
            return

        err = validate_ref_audio(ref_audio_path)
        if err:
            yield None, f"⚠️ {err}"
            return

        yield None, "⏳ Preparing model..."
        m = get_model()

        yield None, "⏳ Running inference..."

        gen_kwargs = dict(
            text=text,
            exaggeration=float(exaggeration),
            cfg_weight=float(cfg_weight),
        )
        if ref_audio_path:
            gen_kwargs["audio_prompt_path"] = ref_audio_path

        t1 = time.time()
        with torch.inference_mode():
            wav = m.generate(**gen_kwargs)
        t2 = time.time()

        if not isinstance(wav, torch.Tensor):
            wav = torch.tensor(wav)

        wav = wav.detach().cpu().to(torch.float32)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)

        out_path = make_unique_wav_path()
        ta.save(out_path, wav, m.sr)
        t3 = time.time()

        yield out_path, (
            f"✅ Done | gen: {t2-t1:.2f}s | save: {t3-t2:.2f}s | total: {t3-t0:.2f}s | "
            f"shape: {tuple(wav.shape)} | sr: {m.sr}"
        )

    except Exception as e:
        yield None, f"❌ Error: {e}\n\n{traceback.format_exc()}"


def get_selected_audio_path(emotion_choice, base, happy, disgust, angry, sleep):
    """
    Pick the correct uploaded file path for the selected emotion.
    Falls back to base if the specific emotion clip is missing.
    """
    selected = base

    if "Happy" in emotion_choice:
        selected = happy or base
    elif "Disgust" in emotion_choice:
        selected = disgust or base
    elif "Angry" in emotion_choice:
        selected = angry or base
    elif "Sleepy" in emotion_choice:
        selected = sleep or base

    return first_path(selected)


def generate_speech_wrapper(
    text,
    emotion_choice,
    base,
    happy,
    disgust,
    angry,
    sleep,
    exaggeration,
    cfg_weight,
):
    """
    Wrapper around generate_speech that handles audio routing.
    """
    base_path = first_path(base)

    if not base_path:
        yield None, "⚠️ Please upload a Base / Neutral voice clip first."
        return

    ref_audio_path = get_selected_audio_path(
        emotion_choice, base, happy, disgust, angry, sleep
    )

    # Safety fallback
    if not ref_audio_path:
        ref_audio_path = base_path

    yield from generate_speech(text, ref_audio_path, exaggeration, cfg_weight)


def backchannel_fn(tag):
    """
    Wrap backchannel buttons so they accept the same shared inputs.
    """
    def _fn(emotion_choice, base, happy, disgust, angry, sleep, exaggeration, cfg_weight):
        yield from generate_speech_wrapper(
            tag, emotion_choice, base, happy, disgust, angry, sleep, exaggeration, cfg_weight
        )
    return _fn


# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(title="Expressive AAC Interface", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ Expressive Voice Interface for AAC")


    
    with gr.Row():
        # Left column: main TTS controls
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Message")
            text_input = gr.Textbox(
                label="",
                placeholder="Type your message here...",
                lines=4,
            )
            speak_btn = gr.Button("🗣️ Speak Message", variant="primary", size="lg")

            gr.Markdown("### 🎭 Emotion Selection")
            emotion_radio = gr.Radio(
                choices=["Neutral 😐", "Happy 😄", "Disgust 🤢", "Angry 😡", "Sleepy 🥱"],
                value="Neutral 😐",
                label="",
                show_label=False,
            )

            # Voice controls hidden; keep fixed defaults
            exaggeration_slider = gr.State(0.5)
            cfg_slider = gr.State(0.5)

            gr.Markdown("### 🎤 Voice Reference Clips")
            base_ref = gr.File(
                label="🎤 Base Voice / Neutral (Required) — Upload WAV file",
                file_count="single",
                file_types=["audio"],
                type="filepath",
            )

            gr.Markdown("📂 Emotion Reference Clips")
            gr.Markdown(
                "Upload 5–10s clips of the same speaker displaying different emotions. "
                "Note: an emotion reference clip is required if you want that specific emotion."
            )

            with gr.Group():
                with gr.Row():
                    happy_ref = gr.File(
                        label="😄 Happy",
                        file_count="single",
                        file_types=["audio"],
                        type="filepath",
                    )
                    angry_ref = gr.File(
                        label="😡 Angry",
                        file_count="single",
                        file_types=["audio"],
                        type="filepath",
                    )

                with gr.Row():
                    disgust_ref = gr.File(
                        label="🤢 Disgust",
                        file_count="single",
                        file_types=["audio"],
                        type="filepath",
                    )
                    sleep_ref = gr.File(
                        label="🥱 Sleepy",
                        file_count="single",
                        file_types=["audio"],
                        type="filepath",
                    )

        # Right column: backchannels
        with gr.Column(scale=1):
            gr.Markdown("### 🔊 Backchannels")

            bc_okay = gr.Button("Okay")
            bc_right = gr.Button("Right")
            bc_yes = gr.Button("Yes")
            bc_no = gr.Button("No")
            bc_tr = gr.Button("That's right")
            bc_uh = gr.Button("Uhhuh")
            bc_yeah = gr.Button("Yeah")
            bc_oh = gr.Button("Oh")
            bc_umm = gr.Button("Umm")

    with gr.Row():
        with gr.Column(scale=2):
            audio_output = gr.Audio(
                label="Generated Audio",
                autoplay=True,
                show_download_button=True,
            )
        with gr.Column(scale=1):
            status_box = gr.Textbox(
                label="Status",
                value="Ready. Upload base refs, choose emotion, and speak.",
                lines=6,
                interactive=False,
            )

    # Keep this order aligned with generate_speech_wrapper():
    # base, happy, disgust, angry, sleep
    all_audio_refs = [base_ref, happy_ref, disgust_ref, angry_ref, sleep_ref]
    shared_inputs = [emotion_radio] + all_audio_refs + [exaggeration_slider, cfg_slider]
    shared_outputs = [audio_output, status_box]

    speak_btn.click(
        fn=generate_speech_wrapper,
        inputs=[text_input] + shared_inputs,
        outputs=shared_outputs,
    )

    bc_okay.click(fn=backchannel_fn("Okay"), inputs=shared_inputs, outputs=shared_outputs)
    bc_right.click(fn=backchannel_fn("Right"), inputs=shared_inputs, outputs=shared_outputs)
    bc_yes.click(fn=backchannel_fn("Yes"), inputs=shared_inputs, outputs=shared_outputs)
    bc_tr.click(fn=backchannel_fn("That's right"), inputs=shared_inputs, outputs=shared_outputs)
    bc_no.click(fn=backchannel_fn("No"), inputs=shared_inputs, outputs=shared_outputs)
    bc_uh.click(fn=backchannel_fn("Uhhuh"), inputs=shared_inputs, outputs=shared_outputs)
    bc_yeah.click(fn=backchannel_fn("Yeah"), inputs=shared_inputs, outputs=shared_outputs)
    bc_oh.click(fn=backchannel_fn("Oh"), inputs=shared_inputs, outputs=shared_outputs)
    bc_umm.click(fn=backchannel_fn("Umm"), inputs=shared_inputs, outputs=shared_outputs)


demo.queue(default_concurrency_limit=1, max_size=20)

if __name__ == "__main__":
    demo.launch()
