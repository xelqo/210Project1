"""Gradio interface for the UCR course-difficulty RAG pipeline (Milestone 5).

Wires the full pipeline together: a question goes through retrieval -> grounded
generation -> an answer with a programmatically-attached Sources block, plus an
expandable view of the exact retrieved chunks so grounding is auditable.

Run:  python src/app.py     # then open the printed local URL
"""

from __future__ import annotations

import gradio as gr

import generate as G
import retrieve as R

EXAMPLES = [
    "What is the average difficulty rating of CS010A?",
    "What is the average difficulty of BIOL005B, and why do students consider it hard?",
    "Which professor do reviewers repeatedly warn against taking for CHEM001A?",
    "According to reviews, what makes CS061 difficult to pass?",
    "Which professor is recommended as the easiest for CS008?",
]


def answer_question(query: str, k: int, use_filter: bool):
    query = (query or "").strip()
    if not query:
        return "Please enter a question.", ""

    out = G.generate(query, k=int(k), use_class_filter=use_filter)

    filt = out["class_filter"] or "none (semantic search across all classes)"
    answer_md = f"**Class filter applied:** {filt}\n\n{out['answer_with_sources']}"

    # Show the exact retrieved chunks so a reader can verify the answer is
    # grounded in them and nothing else.
    chunk_md = ["### Retrieved context (what the model was allowed to use)"]
    for i, r in enumerate(out["results"], 1):
        body = r["text"].split("\n", 1)[1] if "\n" in r["text"] else r["text"]
        chunk_md.append(
            f"**[S{i}] {r['id']}**  ·  cosine distance `{r['distance']:.3f}`\n\n"
            f"> {body}"
        )
    return answer_md, "\n\n".join(chunk_md)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="UCR Course Difficulty — Unofficial Guide") as demo:
        gr.Markdown(
            "# UCR Course Difficulty — The Unofficial Guide\n"
            "Ask about how hard a class is, why, or which professor to take. "
            "Answers come **only** from student reviews retrieved from the "
            "database — if the reviews don't cover it, the system says so."
        )
        with gr.Row():
            # Single-line so pressing Enter submits; multi-line textareas swallow
            # Enter as a newline and the .submit() event never fires.
            query = gr.Textbox(label="Your question", lines=1, scale=4,
                               interactive=True, autofocus=True, submit_btn=True,
                               placeholder="e.g. why is BIOL005B so hard?  (press Enter or click Ask)")
            ask = gr.Button("Ask", variant="primary", scale=1)
        with gr.Row():
            k = gr.Slider(1, 10, value=R.TOP_K, step=1, label="Chunks to retrieve (top-k)")
            use_filter = gr.Checkbox(value=True, label="Filter by class code when named")

        answer = gr.Markdown(value="*Ask a question to see an answer here.*")
        with gr.Accordion("Show retrieved chunks", open=False):
            chunks = gr.Markdown()

        gr.Examples(examples=EXAMPLES, inputs=query, label="Example questions (click to fill)")

        # Wire every way of submitting to the same handler.
        for trigger in (ask.click, query.submit):
            trigger(answer_question, inputs=[query, k, use_filter], outputs=[answer, chunks])
    return demo


if __name__ == "__main__":
    build_ui().launch()
