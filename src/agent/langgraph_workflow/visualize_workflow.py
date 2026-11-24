from pathlib import Path

from src.agent.langgraph_workflow.run_langgraph import build_application


def main():

    print("Building workflow...")

    app = build_application()

    project_root = Path(__file__).resolve().parents[3]
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "insuregraph_workflow.png"

    try:
        graph_image = app.get_graph().draw_mermaid_png()

        with open(output_file, "wb") as file:
            file.write(graph_image)

        print("\nWorkflow diagram created successfully.")
        print("Saved to:", output_file)

    except Exception as error:

        print("\nPNG rendering failed:", error)

        mermaid_code = app.get_graph().draw_mermaid()

        mermaid_file = output_dir / "insuregraph_workflow.mmd"

        mermaid_file.write_text(
            mermaid_code,
            encoding="utf-8"
        )

        print("Mermaid workflow saved to:", mermaid_file)


if __name__ == "__main__":
    main()