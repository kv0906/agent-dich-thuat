from difflib import Differ
from typing import Callable, Optional, Generator
from dataclasses import dataclass
import logging
import time

import docx
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from patch import (
    calculate_chunk_size,
    model_load,
    num_tokens_in_string,
    one_chunk_improve_translation,
    one_chunk_initial_translation,
    one_chunk_reflect_on_translation,
)
from simplemma import simple_tokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class ProgressEvent:
    """Progress event for translation tracking."""
    chunk_index: int
    chunk_total: int
    phase: str  # 'initial', 'reflection', 'improvement'
    phase_index: int  # 1, 2, or 3
    call_index: int
    call_total: int
    partial_translation: str = ""
    message: str = ""


@dataclass 
class PreflightInfo:
    """Preflight estimation info before translation starts."""
    token_count: int
    chunk_count: int
    total_api_calls: int
    estimated_time_min: float
    estimated_time_max: float


class TranslationCancelled(Exception):
    """Raised when translation is cancelled by user."""
    pass


def extract_text(path):
    logger.info(f"📄 Extracting text from: {path}")
    with open(path) as f:
        file_text = f.read()
    logger.info(f"   └── Extracted {len(file_text)} characters")
    return file_text


def extract_pdf(path):
    logger.info(f"📄 Extracting PDF: {path}")
    doc = pymupdf.open(path)
    text = ""
    for i, page in enumerate(doc):
        text += page.get_text()
        if (i + 1) % 10 == 0:
            logger.info(f"   └── Processed {i + 1}/{len(doc)} pages...")
    logger.info(f"   └── Extracted {len(text)} characters from {len(doc)} pages")
    return text


def extract_docx(path):
    logger.info(f"📄 Extracting DOCX: {path}")
    doc = docx.Document(path)
    data = []
    for paragraph in doc.paragraphs:
        data.append(paragraph.text)
    content = "\n\n".join(data)
    logger.info(f"   └── Extracted {len(content)} characters from {len(data)} paragraphs")
    return content


def tokenize(text):
    words = simple_tokenizer(text)
    if " " in text:
        tokens = []
        for word in words:
            tokens.append(word)
            if not word.startswith("'") and not word.endswith("'"):
                tokens.append(" ")
        return tokens[:-1]
    else:
        return words


def diff_texts(text1, text2):
    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)

    d = Differ()
    diff_result = list(d.compare(tokens1, tokens2))

    highlighted_text = []
    for token in diff_result:
        word = token[2:]
        category = None
        if token[0] == "+":
            category = "added"
        elif token[0] == "-":
            category = "removed"
        elif token[0] == "?":
            continue

        highlighted_text.append((word, category))

    return highlighted_text


def get_preflight_info(
    source_text: str,
    max_tokens: int = 1000,
    avg_seconds_per_call: float = 3.0,
) -> PreflightInfo:
    """Get estimation info before starting translation."""
    logger.info("=" * 60)
    logger.info("🔍 PREFLIGHT ANALYSIS")
    logger.info("=" * 60)
    
    num_tokens = num_tokens_in_string(source_text)
    logger.info(f"   📊 Token count: {num_tokens:,}")
    
    if num_tokens < max_tokens:
        chunk_count = 1
        logger.info(f"   📦 Single chunk mode (tokens < {max_tokens})")
    else:
        token_size = calculate_chunk_size(
            token_count=num_tokens, token_limit=max_tokens
        )
        logger.info(f"   📦 Multi-chunk mode, chunk size: {token_size}")
        
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4",
            chunk_size=token_size,
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(source_text)
        chunk_count = len(chunks)
    
    total_api_calls = chunk_count * 3  # 3 phases per chunk
    
    # Estimate time (conservative range)
    estimated_time_min = (total_api_calls * avg_seconds_per_call * 0.8) / 60
    estimated_time_max = (total_api_calls * avg_seconds_per_call * 1.5) / 60
    
    logger.info(f"   📦 Total chunks: {chunk_count:,}")
    logger.info(f"   🔄 Total API calls: {total_api_calls:,}")
    logger.info(f"   ⏱️  Estimated time: {estimated_time_min:.1f} - {estimated_time_max:.1f} minutes")
    logger.info("=" * 60)
    
    return PreflightInfo(
        token_count=num_tokens,
        chunk_count=chunk_count,
        total_api_calls=total_api_calls,
        estimated_time_min=estimated_time_min,
        estimated_time_max=estimated_time_max,
    )


def translator_with_progress(
    source_lang: str,
    target_lang: str,
    source_text: str,
    country: str,
    max_tokens: int = 1000,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Generator[ProgressEvent, None, tuple]:
    """
    Translate with progress events yielded for UI updates.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("🚀 STARTING TRANSLATION")
    logger.info("=" * 60)
    logger.info(f"   📝 Source: {source_lang} → Target: {target_lang}")
    logger.info(f"   🌍 Country/Region: {country}")
    logger.info(f"   📊 Max tokens per chunk: {max_tokens}")
    
    start_time = time.time()
    num_tokens_in_text = num_tokens_in_string(source_text)
    logger.info(f"   📊 Total tokens in text: {num_tokens_in_text:,}")

    if num_tokens_in_text < max_tokens:
        # Single chunk translation
        logger.info("")
        logger.info("📦 MODE: Single Chunk Translation")
        logger.info("-" * 40)
        
        # Phase 1: Initial
        logger.info("")
        logger.info("🔤 PHASE 1/3: Initial Translation")
        yield ProgressEvent(
            chunk_index=1, chunk_total=1,
            phase="initial", phase_index=1,
            call_index=1, call_total=3,
            message="Starting initial translation..."
        )
        
        if cancel_check and cancel_check():
            logger.warning("❌ Translation cancelled by user")
            raise TranslationCancelled()
        
        call_start = time.time()
        init_translation = one_chunk_initial_translation(
            source_lang, target_lang, source_text
        )
        logger.info(f"   ✅ Completed in {time.time() - call_start:.1f}s")
        logger.info(f"   📝 Output length: {len(init_translation)} chars")
        
        # Phase 2: Reflection
        logger.info("")
        logger.info("🤔 PHASE 2/3: Reflection")
        yield ProgressEvent(
            chunk_index=1, chunk_total=1,
            phase="reflection", phase_index=2,
            call_index=2, call_total=3,
            message="Analyzing translation..."
        )
        
        if cancel_check and cancel_check():
            logger.warning("❌ Translation cancelled by user")
            raise TranslationCancelled()
        
        call_start = time.time()
        reflection = one_chunk_reflect_on_translation(
            source_lang, target_lang, source_text, init_translation, country
        )
        logger.info(f"   ✅ Completed in {time.time() - call_start:.1f}s")
        logger.info(f"   📝 Reflection length: {len(reflection)} chars")
        
        # Phase 3: Improvement
        logger.info("")
        logger.info("✨ PHASE 3/3: Improvement")
        yield ProgressEvent(
            chunk_index=1, chunk_total=1,
            phase="improvement", phase_index=3,
            call_index=3, call_total=3,
            message="Improving translation..."
        )
        
        if cancel_check and cancel_check():
            logger.warning("❌ Translation cancelled by user")
            raise TranslationCancelled()
        
        call_start = time.time()
        final_translation = one_chunk_improve_translation(
            source_lang, target_lang, source_text, init_translation, reflection
        )
        logger.info(f"   ✅ Completed in {time.time() - call_start:.1f}s")
        logger.info(f"   📝 Final length: {len(final_translation)} chars")
        
        total_time = time.time() - start_time
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"✅ TRANSLATION COMPLETE in {total_time:.1f}s")
        logger.info("=" * 60)
        
        yield ProgressEvent(
            chunk_index=1, chunk_total=1,
            phase="complete", phase_index=3,
            call_index=3, call_total=3,
            partial_translation=final_translation,
            message="Complete!"
        )
        
        return init_translation, reflection, final_translation

    else:
        # Multi-chunk translation
        logger.info("")
        logger.info("📦 MODE: Multi-Chunk Translation")
        logger.info("-" * 40)

        token_size = calculate_chunk_size(
            token_count=num_tokens_in_text, token_limit=max_tokens
        )
        logger.info(f"   📐 Calculated chunk size: {token_size} tokens")

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4",
            chunk_size=token_size,
            chunk_overlap=0,
        )

        source_text_chunks = text_splitter.split_text(source_text)
        chunk_total = len(source_text_chunks)
        total_calls = chunk_total * 3
        
        logger.info(f"   📦 Split into {chunk_total} chunks")
        logger.info(f"   🔄 Total API calls needed: {total_calls}")
        
        translation_1_chunks = []
        reflection_chunks = []
        translation_2_chunks = []
        
        call_index = 0
        phase_start = time.time()
        
        # Phase 1: Initial translations
        logger.info("")
        logger.info("=" * 60)
        logger.info("🔤 PHASE 1/3: Initial Translations")
        logger.info("=" * 60)
        
        for i, chunk in enumerate(source_text_chunks):
            call_index += 1
            
            logger.info(f"")
            logger.info(f"   📦 Chunk {i + 1}/{chunk_total}")
            logger.info(f"   └── Input: {len(chunk)} chars, ~{num_tokens_in_string(chunk)} tokens")
            
            yield ProgressEvent(
                chunk_index=i + 1, chunk_total=chunk_total,
                phase="initial", phase_index=1,
                call_index=call_index, call_total=total_calls,
                message=f"Initial translation of chunk {i + 1}/{chunk_total}..."
            )
            
            if cancel_check and cancel_check():
                logger.warning("❌ Translation cancelled by user")
                raise TranslationCancelled()
            
            call_start = time.time()
            translation = one_chunk_initial_translation(
                source_lang, target_lang, chunk
            )
            call_duration = time.time() - call_start
            translation_1_chunks.append(translation)
            
            logger.info(f"   └── ✅ Done in {call_duration:.1f}s, output: {len(translation)} chars")
            
            # Progress summary every 10 chunks
            if (i + 1) % 10 == 0:
                elapsed = time.time() - phase_start
                avg_time = elapsed / (i + 1)
                remaining = avg_time * (chunk_total - i - 1)
                logger.info(f"   📊 Progress: {i + 1}/{chunk_total} | Elapsed: {elapsed:.0f}s | ETA: {remaining:.0f}s")
        
        init_translation = "".join(translation_1_chunks)
        phase1_time = time.time() - phase_start
        logger.info(f"")
        logger.info(f"   ✅ Phase 1 complete in {phase1_time:.1f}s")
        
        # Phase 2: Reflections
        phase_start = time.time()
        logger.info("")
        logger.info("=" * 60)
        logger.info("🤔 PHASE 2/3: Reflections")
        logger.info("=" * 60)
        
        for i, (chunk, trans) in enumerate(zip(source_text_chunks, translation_1_chunks)):
            call_index += 1
            
            logger.info(f"")
            logger.info(f"   📦 Chunk {i + 1}/{chunk_total}")
            
            yield ProgressEvent(
                chunk_index=i + 1, chunk_total=chunk_total,
                phase="reflection", phase_index=2,
                call_index=call_index, call_total=total_calls,
                partial_translation=init_translation,
                message=f"Reflecting on chunk {i + 1}/{chunk_total}..."
            )
            
            if cancel_check and cancel_check():
                logger.warning("❌ Translation cancelled by user")
                raise TranslationCancelled()
            
            call_start = time.time()
            reflection = one_chunk_reflect_on_translation(
                source_lang, target_lang, chunk, trans, country
            )
            call_duration = time.time() - call_start
            reflection_chunks.append(reflection)
            
            logger.info(f"   └── ✅ Done in {call_duration:.1f}s")
            
            if (i + 1) % 10 == 0:
                elapsed = time.time() - phase_start
                avg_time = elapsed / (i + 1)
                remaining = avg_time * (chunk_total - i - 1)
                logger.info(f"   📊 Progress: {i + 1}/{chunk_total} | Elapsed: {elapsed:.0f}s | ETA: {remaining:.0f}s")
        
        reflection_text = "\n\n".join(reflection_chunks)
        phase2_time = time.time() - phase_start
        logger.info(f"")
        logger.info(f"   ✅ Phase 2 complete in {phase2_time:.1f}s")
        
        # Phase 3: Improvements
        phase_start = time.time()
        logger.info("")
        logger.info("=" * 60)
        logger.info("✨ PHASE 3/3: Improvements")
        logger.info("=" * 60)
        
        partial_final = ""
        for i, (chunk, trans, refl) in enumerate(zip(
            source_text_chunks, translation_1_chunks, reflection_chunks
        )):
            call_index += 1
            
            logger.info(f"")
            logger.info(f"   📦 Chunk {i + 1}/{chunk_total}")
            
            yield ProgressEvent(
                chunk_index=i + 1, chunk_total=chunk_total,
                phase="improvement", phase_index=3,
                call_index=call_index, call_total=total_calls,
                partial_translation=partial_final,
                message=f"Improving chunk {i + 1}/{chunk_total}..."
            )
            
            if cancel_check and cancel_check():
                logger.warning("❌ Translation cancelled by user")
                raise TranslationCancelled()
            
            call_start = time.time()
            improved = one_chunk_improve_translation(
                source_lang, target_lang, chunk, trans, refl
            )
            call_duration = time.time() - call_start
            translation_2_chunks.append(improved)
            partial_final = "".join(translation_2_chunks)
            
            logger.info(f"   └── ✅ Done in {call_duration:.1f}s, output: {len(improved)} chars")
            
            if (i + 1) % 10 == 0:
                elapsed = time.time() - phase_start
                avg_time = elapsed / (i + 1)
                remaining = avg_time * (chunk_total - i - 1)
                logger.info(f"   📊 Progress: {i + 1}/{chunk_total} | Elapsed: {elapsed:.0f}s | ETA: {remaining:.0f}s")
        
        final_translation = "".join(translation_2_chunks)
        phase3_time = time.time() - phase_start
        logger.info(f"")
        logger.info(f"   ✅ Phase 3 complete in {phase3_time:.1f}s")
        
        total_time = time.time() - start_time
        logger.info("")
        logger.info("=" * 60)
        logger.info("✅ TRANSLATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"   ⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        logger.info(f"   📊 Phase 1 (Initial): {phase1_time:.1f}s")
        logger.info(f"   📊 Phase 2 (Reflection): {phase2_time:.1f}s")
        logger.info(f"   📊 Phase 3 (Improvement): {phase3_time:.1f}s")
        logger.info(f"   📝 Final output: {len(final_translation):,} chars")
        logger.info("=" * 60)
        
        yield ProgressEvent(
            chunk_index=chunk_total, chunk_total=chunk_total,
            phase="complete", phase_index=3,
            call_index=total_calls, call_total=total_calls,
            partial_translation=final_translation,
            message="Translation complete!"
        )
        
        return init_translation, reflection_text, final_translation


def translator(
    source_lang: str,
    target_lang: str,
    source_text: str,
    country: str,
    max_tokens: int = 1000,
):
    """
    Simple translator without progress (backwards compatible).
    """
    result = None
    gen = translator_with_progress(
        source_lang, target_lang, source_text, country, max_tokens
    )
    
    try:
        while True:
            event = next(gen)
    except StopIteration as e:
        result = e.value
    
    return result
