
import sys
import json
import os
import time
from pathlib import Path

# Add project root to Python import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd

from dotenv import load_dotenv

from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import chunk_documents
from src.embeddings.embedding_model import EmbeddingModel
from src.retrieval.vector_search import VectorSearch
from src.extraction.entity_extractor import EntityExtractor, ExtractionResult
from src.knowledge_graph.graph_builder import KnowledgeGraphBuilder
from src.knowledge_graph.graph_query import GraphQuery
from src.agent.rag_agent import RAGAgent
from src.agent.model_manager import LazyModelLoader


DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
UPLOAD_DIR = PROJECT_ROOT / "outputs" / "dashboard_uploads"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

load_dotenv(PROJECT_ROOT / ".env")

st.set_page_config(
    page_title="InsureGraph AI",
    page_icon="🛡️",
    layout="wide"
)


# --------------------------------------------------
# 1. BUILD APPLICATION
# --------------------------------------------------

def build_application(pdf_paths, progress):

    documents = []

    for index, pdf_path in enumerate(pdf_paths):

        documents.extend(load_pdf(str(pdf_path)))

        progress.progress(
            (index + 1) / len(pdf_paths),
            text=f"Reading {pdf_path.name}"
        )

    if not documents:
        raise ValueError("No readable PDF content was found.")

    chunks = chunk_documents(documents)

    if not chunks:
        raise ValueError("No document chunks were created.")

    # Build embeddings and vector index.
    embedding_model = EmbeddingModel()

    vector_search = VectorSearch(embedding_model)
    vector_search.index_documents(chunks)

    # Extract entities and relationships.
    extractor = EntityExtractor()

    entities = {}
    relationships = {}

    for chunk in chunks:

        result = extractor.extract(chunk.page_content)

        for entity in result.entities:
            entities[(entity.name, entity.entity_type)] = entity

        for relationship in result.relationships:

            key = (
                relationship.source,
                relationship.relation,
                relationship.target
            )

            relationships[key] = relationship

    extraction = ExtractionResult(
        entities=list(entities.values()),
        relationships=list(relationships.values())
    )

    # Build knowledge graph.
    graph_builder = KnowledgeGraphBuilder()
    graph_builder.build(extraction)

    graph_query = GraphQuery(graph_builder)

    # Configure lazy local LLM loading.
    model_name = os.getenv(
        "HF_LLM_MODEL",
        "Qwen/Qwen2.5-0.5B-Instruct"
    )

    model_loader = LazyModelLoader(model_name)

    # Create RAG agent.
    agent = RAGAgent(
        vector_search,
        graph_query
    )

    progress.progress(
        1.0,
        text="Application is ready."
    )

    return {
        "agent": agent,
        "model_loader": model_loader,
        "documents": documents,
        "chunks": chunks,
        "entities": list(entities.values()),
        "relationships": list(relationships.values()),
        "graph": graph_builder.get_graph_data(),
        "pdf_names": [path.name for path in pdf_paths]
    }


# --------------------------------------------------
# 2. LOAD EVALUATION METRICS
# --------------------------------------------------

def load_metrics(filename):

    file_path = OUTPUT_DIR / filename

    if not file_path.exists():
        return None

    try:
        with file_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    except (json.JSONDecodeError, OSError):
        return None


# --------------------------------------------------
# 3. KNOWLEDGE GRAPH VISUALIZATION
# --------------------------------------------------

def make_graphviz(graph_data):

    dot = ["digraph InsureGraph {"]
    dot.append('rankdir="LR";')
    dot.append(
        'node [shape=box, style="rounded,filled", '
        'fillcolor="#E8F0FE"];'
    )

    for node in graph_data["nodes"]:

        name = (
            str(node["id"])
            .replace("\\", "\\\\")
            .replace('"', '\\"')
        )

        dot.append(f'"{name}";')

    for edge in graph_data["edges"]:

        source = (
            str(edge["source"])
            .replace("\\", "\\\\")
            .replace('"', '\\"')
        )

        target = (
            str(edge["target"])
            .replace("\\", "\\\\")
            .replace('"', '\\"')
        )

        relation = str(edge["relation"]).replace('"', '\\"')

        dot.append(
            f'"{source}" -> "{target}" [label="{relation}"];'
        )

    dot.append("}")

    return "\n".join(dot)


# --------------------------------------------------
# 4. SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("🛡️ InsureGraph AI")
st.sidebar.caption("Local Insurance Intelligence Platform")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Document Ingestion",
        "Ask Questions",
        "Knowledge Graph",
        "Evaluation"
    ]
)

st.title("🛡️ InsureGraph AI")
st.caption(
    "AI-powered insurance document intelligence "
    "using RAG, Hugging Face and Knowledge Graphs"
)


# --------------------------------------------------
# 5. SESSION STATE
# --------------------------------------------------

if "application" not in st.session_state:
    st.session_state.application = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

application = st.session_state.application


# --------------------------------------------------
# 6. DOCUMENT INGESTION
# --------------------------------------------------

if page == "Document Ingestion":

    st.header("📄 Document Ingestion")

    st.write(
        "Upload one or more insurance policy PDFs "
        "to build your searchable insurance knowledge base."
    )

    uploaded_files = st.file_uploader(
        "Upload insurance PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Process Documents", type="primary"):

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        for uploaded_file in uploaded_files or []:

            safe_name = Path(uploaded_file.name).name

            destination = UPLOAD_DIR / safe_name

            destination.write_bytes(
                uploaded_file.getvalue()
            )

        pdf_paths = sorted(
            list(DOCUMENTS_DIR.glob("*.pdf"))
            + list(UPLOAD_DIR.glob("*.pdf"))
        )

        # Remove duplicate paths.
        pdf_paths = list({
            path.resolve(): path
            for path in pdf_paths
        }.values())

        if not pdf_paths:

            st.error("No PDF documents found.")

        else:

            try:

                progress = st.progress(
                    0,
                    text="Starting ingestion..."
                )

                with st.spinner(
                    "Building InsureGraph AI..."
                ):

                    new_application = build_application(
                        pdf_paths,
                        progress
                    )

                st.session_state.application = new_application
                st.session_state.chat_history = []

                application = new_application

                st.success(
                    "Documents processed successfully!"
                )

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Documents",
                    len(application["pdf_names"])
                )

                col2.metric(
                    "Chunks",
                    len(application["chunks"])
                )

                col3.metric(
                    "Entities",
                    len(application["entities"])
                )

                col4.metric(
                    "Relationships",
                    len(application["relationships"])
                )

                st.info(
                    "The local language model is configured "
                    "for lazy loading. It will load only when "
                    "a question requires LLM generation."
                )

            except Exception as error:

                st.error(
                    f"Processing failed: {error}"
                )


# Refresh application after ingestion.
application = st.session_state.application


# --------------------------------------------------
# 7. DASHBOARD OVERVIEW
# --------------------------------------------------

if page == "Dashboard":

    st.header("📊 Project Overview")

    if application:

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Documents",
            len(application["pdf_names"])
        )

        col2.metric(
            "Chunks",
            len(application["chunks"])
        )

        col3.metric(
            "Entities",
            len(application["entities"])
        )

        col4.metric(
            "Relationships",
            len(application["relationships"])
        )

        st.subheader("Processed Documents")

        for name in application["pdf_names"]:
            st.write(f"📄 {name}")

        st.subheader("Local Model")

        if application["model_loader"].loaded:
            st.success("Local LLM is loaded.")
        else:
            st.info("Local LLM is not loaded. Graph answering is available.")

    else:

        st.info(
            "Go to Document Ingestion and process your PDF files."
        )

    st.divider()

    st.subheader("Project Modules")

    module_columns = st.columns(3)

    modules = [
        ("PDF Ingestion", "Extract text from insurance PDFs"),
        ("Vector Search", "Retrieve relevant document chunks"),
        ("Knowledge Graph", "Connect insurance entities"),
        ("Local LLM", "Generate answers using local inference"),
        ("RAG Evaluation", "Measure answer quality and grounding"),
        ("Streamlit UI", "Interact with the complete platform")
    ]

    for index, (title, description) in enumerate(modules):

        with module_columns[index % 3]:

            st.markdown(f"**{title}**")
            st.caption(description)


# --------------------------------------------------
# 8. ASK INSURANCE QUESTIONS
# --------------------------------------------------

elif page == "Ask Questions":

    st.header("💬 Ask Your Insurance Documents")

    if not application:

        st.warning(
            "Please process your insurance documents first."
        )

    else:

        st.caption(
            "Ask questions about individual policies, "
            "compare policies, or rank coverage and premiums."
        )

        example_questions = [
            "What is the coverage amount for Rahul Sharma?",
            "Which policy has the highest coverage?",
            "Which policy has the lowest premium?",
            "Rank all policies by coverage.",
            "Compare all policies.",
            "List all insurance providers."
        ]

        with st.expander("Example questions"):

            for example in example_questions:
                st.write(f"• {example}")

        question = st.text_input(
            "Enter your insurance question",
            placeholder="Which policy has the highest coverage?"
        )

        if st.button(
            "Get Answer",
            type="primary"
        ):

            if not question.strip():

                st.warning("Please enter a question.")

            else:

                try:

                    start_time = time.perf_counter()

                    with st.spinner(
                        "Searching documents and verifying evidence..."
                    ):

                        result = application["agent"].answer(
                            question,
                            application["model_loader"],
                            None
                        )

                    total_time = (
                        time.perf_counter() - start_time
                    )

                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": result["answer"],
                        "answer_source": result["answer_source"],
                        "sources": result["sources"],
                        "time": total_time
                    })

                    st.subheader("Answer")

                    st.success(result["answer"])

                    col1, col2 = st.columns(2)

                    col1.metric(
                        "Response Time",
                        f"{total_time:.3f} sec"
                    )

                    col2.metric(
                        "Answer Source",
                        result["answer_source"]
                    )

                    st.subheader("Sources")

                    for source in result["sources"]:

                        st.write(
                            f"📄 {source.get('source', 'Unknown')} "
                            f"| Page: {source.get('page', 'Unknown')}"
                        )

                    st.subheader("Knowledge Graph Evidence")

                    if result["graph_evidence"]:

                        st.dataframe(
                            pd.DataFrame(result["graph_evidence"]),
                            use_container_width=True
                        )

                    else:

                        st.info(
                            "No related graph evidence found."
                        )

                    if application["model_loader"].loaded:

                        st.caption(
                            "Local LLM status: Loaded"
                        )

                    else:

                        st.caption(
                            "Local LLM status: Not loaded"
                        )

                except Exception as error:

                    st.error(
                        f"Question processing failed: {error}"
                    )

        if st.session_state.chat_history:

            st.divider()

            st.subheader("Recent Questions")

            for item in reversed(
                st.session_state.chat_history[-5:]
            ):

                with st.expander(item["question"]):

                    st.write(item["answer"])

                    st.caption(
                        f"Source: {item['answer_source']} | "
                        f"Time: {item['time']:.3f} seconds"
                    )


# --------------------------------------------------
# 9. KNOWLEDGE GRAPH
# --------------------------------------------------

elif page == "Knowledge Graph":

    st.header("🕸️ Insurance Knowledge Graph")

    if not application:

        st.warning("Process your documents first.")

    else:

        st.subheader("Extracted Entities")

        entity_rows = [
            {
                "Name": entity.name,
                "Type": entity.entity_type,
                "Evidence": entity.evidence
            }
            for entity in application["entities"]
        ]

        st.dataframe(
            pd.DataFrame(entity_rows),
            use_container_width=True
        )

        st.subheader("Extracted Relationships")

        relationship_rows = [
            {
                "Source": relationship.source,
                "Relation": relationship.relation,
                "Target": relationship.target
            }
            for relationship in application["relationships"]
        ]

        st.dataframe(
            pd.DataFrame(relationship_rows),
            use_container_width=True
        )

        st.subheader("Graph Visualization")

        graph_data = application["graph"]

        if graph_data["nodes"]:

            st.graphviz_chart(
                make_graphviz(graph_data)
            )

        else:

            st.info("No graph nodes available.")


# --------------------------------------------------
# 10. EVALUATION
# --------------------------------------------------

elif page == "Evaluation":

    st.header("📈 Model Evaluation")

    extraction_metrics = load_metrics(
        "evaluation_results.json"
    )

    rag_metrics = load_metrics(
        "rag_evaluation_results.json"
    )

    if extraction_metrics:

        st.subheader("Entity Extraction")

        entity_metrics = extraction_metrics["entities"]
        relationship_metrics = extraction_metrics["relationships"]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Entity Precision",
            f"{entity_metrics['precision']:.0%}"
        )

        col2.metric(
            "Entity Recall",
            f"{entity_metrics['recall']:.0%}"
        )

        col3.metric(
            "Entity F1-score",
            f"{entity_metrics['f1_score']:.0%}"
        )

        st.subheader("Relationship Extraction")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Precision",
            f"{relationship_metrics['precision']:.0%}"
        )

        col2.metric(
            "Recall",
            f"{relationship_metrics['recall']:.0%}"
        )

        col3.metric(
            "F1-score",
            f"{relationship_metrics['f1_score']:.0%}"
        )

    else:

        st.info(
            "Run the extraction evaluation to display its metrics."
        )

    if rag_metrics:

        st.subheader("RAG Evaluation")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Questions",
            rag_metrics["total_questions"]
        )

        col2.metric(
            "Answer Accuracy",
            f"{rag_metrics['answer_accuracy']:.0%}"
        )

        col3.metric(
            "Evidence Grounding",
            f"{rag_metrics['evidence_grounding_rate']:.0%}"
        )

        st.subheader("Individual Test Results")

        st.dataframe(
            pd.DataFrame(rag_metrics["results"]),
            use_container_width=True
        )

    else:

        st.info(
            "Run the RAG evaluation to display its metrics."
        )
