import os
import re
import time

import streamlit as st

from process import (
    extract_docx,
    extract_pdf,
    extract_text,
    model_load,
    translator_with_progress,
    get_preflight_info,
    TranslationCancelled,
    num_tokens_in_string,
)

st.set_page_config(
    page_title="Translation Agent - DeepSeek",
    page_icon="🌐",
    layout="wide",
)

# Initialize session state
if "translation_started" not in st.session_state:
    st.session_state.translation_started = False
if "cancel_requested" not in st.session_state:
    st.session_state.cancel_requested = False
if "results" not in st.session_state:
    st.session_state.results = None
if "call_times" not in st.session_state:
    st.session_state.call_times = []
if "failed_state" not in st.session_state:
    st.session_state.failed_state = None
if "error_message" not in st.session_state:
    st.session_state.error_message = None
if "translation_params" not in st.session_state:
    st.session_state.translation_params = None


def format_time(minutes: float) -> str:
    """Format minutes to readable string."""
    if minutes < 1:
        return f"{int(minutes * 60)}s"
    elif minutes < 60:
        return f"{int(minutes)}m {int((minutes % 1) * 60)}s"
    else:
        hours = int(minutes // 60)
        mins = int(minutes % 60)
        return f"{hours}h {mins}m"


def request_cancel():
    st.session_state.cancel_requested = True


def check_cancel() -> bool:
    return st.session_state.cancel_requested


st.title("🌐 Translation Agent - DeepSeek")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input(
        "DeepSeek API Key",
        type="password",
        placeholder="Leave empty to use env var",
        help="Uses DEEPSEEK_API_KEY from environment if empty",
    )
    
    st.subheader("Languages")
    col1, col2 = st.columns(2)
    with col1:
        source_lang = st.text_input("From", value="Chinese")
    with col2:
        target_lang = st.text_input("To", value="English")
    
    country = st.text_input("Country/Region", value="USA")
    
    with st.expander("Advanced Options"):
        max_tokens = st.slider("Max Tokens Per Chunk", 512, 2046, 1000, 8)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)

# Main content
st.subheader("📝 Source Text")

uploaded_file = st.file_uploader(
    "Upload a file (optional)", 
    type=["txt", "pdf", "docx", "md", "json", "py", "cpp"],
)

default_text = "今天天气真好，我想去公园散步。"

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1]
    temp_path = f"/tmp/upload.{file_type}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    if file_type == "pdf":
        default_text = extract_pdf(temp_path)
    elif file_type == "docx":
        default_text = extract_docx(temp_path)
    else:
        default_text = uploaded_file.getvalue().decode("utf-8")
    
    default_text = re.sub(r"(?m)^\s*$\n?", "", default_text)

source_text = st.text_area(
    "Enter text to translate",
    value=default_text,
    height=200,
    label_visibility="collapsed",
)

# Preflight estimation
if source_text and source_lang != target_lang:
    clean_text = re.sub(r"(?m)^\s*$\n?", "", source_text)
    preflight = get_preflight_info(clean_text, max_tokens)
    
    st.divider()
    
    # Preflight info panel
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Tokens", f"{preflight.token_count:,}")
    with col2:
        st.metric("📦 Chunks", f"{preflight.chunk_count:,}")
    with col3:
        st.metric("🔄 API Calls", f"{preflight.total_api_calls:,}")
    with col4:
        eta_str = f"{format_time(preflight.estimated_time_min)} - {format_time(preflight.estimated_time_max)}"
        st.metric("⏱️ Est. Time", eta_str)
    
    st.divider()

# Translation controls
has_failed_state = st.session_state.failed_state is not None

if has_failed_state:
    col_resume, col_start, col_cancel = st.columns([1, 2, 1])
    
    with col_resume:
        resume_button = st.button(
            "🔄 Resume",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.translation_started,
            help="Resume translation from where it failed",
        )
    
    with col_start:
        start_button = st.button(
            "🚀 Start Fresh", 
            use_container_width=True,
            disabled=st.session_state.translation_started,
        )
    
    with col_cancel:
        cancel_button = st.button(
            "⏹️ Cancel",
            use_container_width=True,
            disabled=not st.session_state.translation_started,
            on_click=request_cancel,
        )
else:
    resume_button = False
    col_start, col_cancel = st.columns([3, 1])
    
    with col_start:
        start_button = st.button(
            "🚀 Start Translation", 
            type="primary", 
            use_container_width=True,
            disabled=st.session_state.translation_started,
        )
    
    with col_cancel:
        cancel_button = st.button(
            "⏹️ Cancel",
            use_container_width=True,
            disabled=not st.session_state.translation_started,
            on_click=request_cancel,
        )

# Show error message if translation failed
if st.session_state.error_message and not st.session_state.translation_started:
    with st.container():
        st.error(f"❌ {st.session_state.error_message}")
        if st.session_state.failed_state:
            state = st.session_state.failed_state
            completed = 0
            if state.current_phase == "initial":
                completed = len(state.translation_1_chunks)
            elif state.current_phase == "reflection":
                completed = len(state.source_text_chunks) + len(state.reflection_chunks)
            else:
                completed = len(state.source_text_chunks) * 2 + len(state.translation_2_chunks)
            total = len(state.source_text_chunks) * 3
            st.info(f"💾 Progress saved: {completed}/{total} API calls completed. Click **Resume** to continue.")
        
        if st.button("🗑️ Clear Error", use_container_width=True):
            st.session_state.error_message = None
            st.session_state.failed_state = None
            st.rerun()

# Progress display area
progress_container = st.container()

# Results display
st.subheader("📄 Translation Result")
result_placeholder = st.empty()
partial_output = st.empty()

with st.expander("📋 Details", expanded=False):
    detail_init = st.empty()
    detail_reflection = st.empty()

# Display previous results if available
if st.session_state.results and not st.session_state.translation_started:
    init, reflection, final = st.session_state.results
    result_placeholder.text_area(
        "Final Translation",
        value=final,
        height=300,
        label_visibility="collapsed",
    )
    detail_init.text_area("Initial Translation", value=init, height=150)
    detail_reflection.text_area("Reflection", value=reflection, height=150)

# Run translation
should_start = start_button or resume_button
is_resume = resume_button and st.session_state.failed_state is not None

if should_start:
    if not source_text:
        st.error("Please enter text to translate")
    elif source_lang == target_lang:
        st.error("Source and target languages must be different")
    else:
        st.session_state.translation_started = True
        st.session_state.cancel_requested = False
        
        # Only reset state if starting fresh
        if not is_resume:
            st.session_state.results = None
            st.session_state.call_times = []
            st.session_state.failed_state = None
            st.session_state.error_message = None
        
        try:
            # Load model
            with st.spinner("Loading DeepSeek model..."):
                model_load(api_key=api_key, temperature=temperature)
            
            clean_text = re.sub(r"(?m)^\s*$\n?", "", source_text)
            preflight = get_preflight_info(clean_text, max_tokens)
            
            # Store params for resume
            st.session_state.translation_params = {
                "source_lang": source_lang,
                "target_lang": target_lang,
                "source_text": clean_text,
                "country": country,
                "max_tokens": max_tokens,
            }
            
            # Set up progress UI
            with progress_container:
                progress_bar = st.progress(0, text="Starting translation...")
                status_text = st.empty()
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                with metrics_col1:
                    eta_metric = st.empty()
                with metrics_col2:
                    speed_metric = st.empty()
                with metrics_col3:
                    elapsed_metric = st.empty()
            
            # Run translation with progress
            start_time = time.time()
            
            # Get resume state if resuming
            resume_state = st.session_state.failed_state if is_resume else None
            if is_resume:
                status_text.markdown("🔄 **Resuming from saved state...**")
            
            gen = translator_with_progress(
                source_lang=source_lang,
                target_lang=target_lang,
                source_text=clean_text,
                country=country,
                max_tokens=max_tokens,
                cancel_check=check_cancel,
                resume_state=resume_state,
            )
            
            final_result = None
            
            try:
                while True:
                    call_start = time.time()
                    event = next(gen)
                    call_duration = time.time() - call_start
                    
                    # Track call times for ETA
                    if call_duration > 0.1:  # Only track actual API calls
                        st.session_state.call_times.append(call_duration)
                    
                    # Calculate progress
                    progress = event.call_index / event.call_total
                    progress_bar.progress(progress, text=event.message)
                    
                    # Update status
                    phase_names = {
                        "initial": "🔤 Initial Translation",
                        "reflection": "🤔 Reflection", 
                        "improvement": "✨ Improvement",
                        "complete": "✅ Complete"
                    }
                    phase_name = phase_names.get(event.phase, event.phase)
                    
                    status_text.markdown(
                        f"**Chunk {event.chunk_index}/{event.chunk_total}** • "
                        f"**Phase:** {phase_name} • "
                        f"**Call:** {event.call_index}/{event.call_total}"
                    )
                    
                    # Calculate ETA
                    elapsed = time.time() - start_time
                    elapsed_metric.metric("⏱️ Elapsed", format_time(elapsed / 60))
                    
                    if st.session_state.call_times:
                        avg_call_time = sum(st.session_state.call_times) / len(st.session_state.call_times)
                        remaining_calls = event.call_total - event.call_index
                        eta_seconds = remaining_calls * avg_call_time
                        eta_metric.metric("🎯 ETA", format_time(eta_seconds / 60))
                        
                        calls_per_min = 60 / avg_call_time if avg_call_time > 0 else 0
                        speed_metric.metric("⚡ Speed", f"{calls_per_min:.1f} calls/min")
                    
                    # Show partial output
                    if event.partial_translation:
                        partial_output.text_area(
                            "Partial Translation (live)",
                            value=event.partial_translation[:2000] + ("..." if len(event.partial_translation) > 2000 else ""),
                            height=150,
                            label_visibility="collapsed",
                        )
                    
            except StopIteration as e:
                final_result = e.value
                # Clear failed state on success
                st.session_state.failed_state = None
                st.session_state.error_message = None
            except TranslationCancelled:
                st.warning("⚠️ Translation cancelled by user")
                final_result = None
            except TranslationFailed as e:
                st.session_state.failed_state = e.state
                st.session_state.error_message = str(e)
                final_result = None
            
            # Show final results
            if final_result:
                init, reflection, final = final_result
                st.session_state.results = final_result
                
                progress_bar.progress(1.0, text="✅ Translation complete!")
                
                result_placeholder.text_area(
                    "Final Translation",
                    value=final,
                    height=300,
                    label_visibility="collapsed",
                )
                
                detail_init.text_area("Initial Translation", value=init, height=150)
                detail_reflection.text_area("Reflection", value=reflection, height=150)
                
                # Clear partial output
                partial_output.empty()
                
                total_time = time.time() - start_time
                st.success(f"✅ Translation completed in {format_time(total_time / 60)}!")
        
        except Exception as e:
            st.session_state.error_message = str(e)
            st.error(f"❌ Error: {e}")
        
        finally:
            st.session_state.translation_started = False
            st.session_state.cancel_requested = False
