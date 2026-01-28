"""
Script to visualize the LangGraph workflow.
Requires: pip install pygraphviz (optional)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.workflow import FeatureDiscoveryWorkflow


def main():
    """Generate workflow visualization."""
    
    print("Initializing workflow...")
    workflow = FeatureDiscoveryWorkflow(max_iterations=3)
    
    try:
        print("Generating visualization...")
        
        # Try to get the mermaid diagram
        mermaid = workflow.graph.get_graph().draw_mermaid()
        
        print("\nWorkflow Diagram (Mermaid):")
        print("="*80)
        print(mermaid)
        print("="*80)
        
        # Save to file
        output_file = "workflow_diagram.md"
        with open(output_file, 'w') as f:
            f.write("# Feature Discovery Workflow\n\n")
            f.write("```mermaid\n")
            f.write(mermaid)
            f.write("\n```\n")
        
        print(f"\n✅ Diagram saved to {output_file}")
        print("\nYou can visualize this in:")
        print("- GitHub/GitLab (renders mermaid automatically)")
        print("- VS Code (with Markdown Preview Mermaid Support extension)")
        print("- Online: https://mermaid.live/")
        
    except Exception as e:
        print(f"Error generating visualization: {e}")
        print("\nWorkflow structure:")
        print("  START → GENERATE → EVALUATE → (continue loop or) → FINALIZE → END")


if __name__ == "__main__":
    main()
