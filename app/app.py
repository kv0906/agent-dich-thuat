import os
import re

import gradio as gr
from process import (
    extract_docx,
    extract_pdf,
    extract_text,
    model_load,
    translator,
)


def translate_text(
    api_key: str,
    source_lang: str,
    target_lang: str,
    source_text: str,
    country: str,
    max_tokens: int,
    temperature: float,
):
    if not source_text or source_lang == target_lang:
        raise gr.Error(
            "Please check that the content or options are entered correctly."
        )

    try:
        model_load(api_key=api_key, temperature=temperature)
    except Exception as e:
        raise gr.Error(f"An unexpected error occurred: {e}") from e

    source_text = re.sub(r"(?m)^\s*$\n?", "", source_text)

    init_translation, reflect_translation, final_translation = translator(
        source_lang=source_lang,
        target_lang=target_lang,
        source_text=source_text,
        country=country,
        max_tokens=max_tokens,
    )

    return init_translation, reflect_translation, final_translation


def read_doc(path):
    file_type = path.split(".")[-1]
    if file_type in ["pdf", "txt", "py", "docx", "json", "cpp", "md"]:
        if file_type.endswith("pdf"):
            content = extract_pdf(path)
        elif file_type.endswith("docx"):
            content = extract_docx(path)
        else:
            content = extract_text(path)
        return re.sub(r"(?m)^\s*$\n?", "", content)
    else:
        raise gr.Error("Oops, unsupported files.")


def switch(source_lang, source_text, target_lang, output_final):
    if output_final:
        return target_lang, output_final, source_lang, source_text
    else:
        return target_lang, source_text, source_lang, ""


TITLE = """
<div style="text-align: center;">
    <h1 style="color: #6366f1">Translation Agent - DeepSeek</h1>
</div>
"""

with gr.Blocks(theme="soft") as demo:
    gr.HTML(TITLE)
    
    with gr.Row():
        with gr.Column(scale=1):
            api_key = gr.Textbox(
                label="DeepSeek API Key",
                type="password",
                placeholder="Leave empty to use DEEPSEEK_API_KEY env var",
            )
            with gr.Row():
                source_lang = gr.Textbox(label="Source Lang", value="Chinese")
                target_lang = gr.Textbox(label="Target Lang", value="English")
            switch_btn = gr.Button(value="🔄️ Swap")
            country = gr.Textbox(label="Country", value="USA")
            
            with gr.Accordion("Advanced Options", open=False):
                max_tokens = gr.Slider(
                    label="Max tokens Per Chunk",
                    minimum=512,
                    maximum=2046,
                    value=1000,
                    step=8,
                )
                temperature = gr.Slider(
                    label="Temperature",
                    minimum=0,
                    maximum=1.0,
                    value=0.3,
                    step=0.1,
                )

        with gr.Column(scale=3):
            source_text = gr.Textbox(
                label="Source Text",
                value="今天天气真好，我想去公园散步。",
                lines=8,
            )
            output_final = gr.Textbox(
                label="Final Translation", 
                lines=8, 
                show_copy_button=True
            )
            
            with gr.Accordion("Details", open=False):
                output_init = gr.Textbox(
                    label="Initial Translation", 
                    lines=4, 
                    show_copy_button=True
                )
                output_reflect = gr.Textbox(
                    label="Reflection", 
                    lines=4, 
                    show_copy_button=True
                )

    with gr.Row():
        submit = gr.Button(value="Translate", variant="primary")
        upload = gr.UploadButton(label="Upload File", file_types=["text"])
        clear = gr.ClearButton([source_text, output_init, output_reflect, output_final])

    switch_btn.click(
        fn=switch,
        inputs=[source_lang, source_text, target_lang, output_final],
        outputs=[source_lang, source_text, target_lang, output_final],
    )

    submit.click(
        fn=translate_text,
        inputs=[
            api_key,
            source_lang,
            target_lang,
            source_text,
            country,
            max_tokens,
            temperature,
        ],
        outputs=[output_init, output_reflect, output_final],
    )
    
    upload.upload(fn=read_doc, inputs=upload, outputs=source_text)

if __name__ == "__main__":
    demo.launch()
