"""
Basic example of using the Feature Discovery System.
"""
import os
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.workflow import FeatureDiscoveryWorkflow


def print_results(result: dict):
    """Pretty print the results."""
    
    print("\n" + "="*80)
    print("FEATURE DISCOVERY RESULTS")
    print("="*80)
    
    print(f"\nTotal Iterations: {result['total_iterations']}")
    
    print("\n" + "-"*80)
    print("EVALUATION RUBRIC")
    print("-"*80)
    rubric = result['rubric']
    print(f"\nRationale: {rubric['rationale']}\n")
    print("Criteria:")
    for criterion in rubric['criteria']:
        print(f"  • {criterion['name']} (weight: {criterion['weight']:.2f})")
        print(f"    {criterion['description']}")
    
    print("\n" + "-"*80)
    print("GENERATED FEATURES")
    print("-"*80)
    print(f"\nTaxonomy: {result['taxonomy_explanation']}\n")
    
    features = result['features']
    evaluations = {e['feature_name']: e for e in result['evaluations']}
    
    for feature in features:
        eval_data = evaluations.get(feature['name'], {})
        overall_score = eval_data.get('overall_score', 0)
        
        print(f"\n📊 {feature['name']} (Score: {overall_score:.2f}/10)")
        print(f"   Type: {feature['data_type']}")
        print(f"   Taxonomy: {feature['taxonomy']}")
        print(f"   Description: {feature['description']}")
        print(f"   Rationale: {feature['rationale']}")
        
        if eval_data:
            print(f"   Feedback: {eval_data.get('feedback', 'N/A')}")


def main():
    """Run the basic example."""
    
    # Example business problem
    business_problem = """
    Predict customer churn in retail banking. We want to identify customers 
    likely to close their accounts in the next 90 days. This will help our 
    retention team proactively reach out with targeted offers and support.
    """
    
    print("Initializing Feature Discovery Workflow...")
    print(f"Business Problem: {business_problem.strip()}\n")
    
    # Initialize workflow
    workflow = FeatureDiscoveryWorkflow(
        llm_provider="anthropic",  # or "openai"
        model_name="claude-sonnet-4-5-20250929",  # or "gpt-4-turbo-preview"
        max_iterations=3,
        temperature=0.7
    )
    
    # Run workflow
    print("Running workflow (this may take 1-2 minutes)...\n")
    result = workflow.run(business_problem, verbose=True)
    
    # Print results
    print_results(result)
    
    # Optionally save to file
    output_file = "feature_discovery_output.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n✅ Results saved to {output_file}")


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("Error: No API key found!")
        print("Please set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable")
        print("Or create a .env file with your API key")
        sys.exit(1)
    
    main()
