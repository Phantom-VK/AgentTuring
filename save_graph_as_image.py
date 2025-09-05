from agentturing.pipelines.main_pipeline import build_graph

if __name__ == "__main__":
    graph = build_graph()

    # Get PNG bytes
    png_bytes = graph.get_graph().draw_mermaid_png()

    # Save to file
    with open("pipeline_graph.png", "wb") as f:
        f.write(png_bytes)

    print("Graph image saved as pipeline_graph.png")
