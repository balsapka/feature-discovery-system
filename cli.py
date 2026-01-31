#!/usr/bin/env python3
"""
Command-line interface for the Feature Discovery System.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.workflow import FeatureDiscoveryWorkflow


def print_banner():
    """Print welcome banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║          Feature Discovery System for Banking AI              ║
║                                                               ║
║  Generate, evaluate, and refine features for ML models       ║
╚═══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def format_features(features, evaluations):
    """Format features for terminal output."""
    output = []
    eval_dict = {e['feature_name']: e for e in evaluations}

    for i, feature in enumerate(features, 1):
        eval_data = eval_dict.get(feature['name'], {})
        score = eval_data.get('overall_score', 0)

        output.append(f"\n{'='*70}")
        output.append(f"Feature #{i}: {feature['name']}")
        output.append(f"{'='*70}")
        output.append(f"Score: {score:.2f}/10")
        output.append(f"Type: {feature['data_type']}")
        output.append(f"Taxonomy: {feature['taxonomy']}")
        output.append(f"\nDescription:")
        output.append(f"  {feature['description']}")
        output.append(f"\nRationale:")
        output.append(f"  {feature['rationale']}")

        if eval_data:
            output.append(f"\nEvaluation Feedback:")
            output.append(f"  {eval_data.get('feedback', 'N/A')}")

    return "\n".join(output)


def format_rubric(rubric):
    """Format rubric for terminal output."""
    output = ["\n" + "="*70]
    output.append("EVALUATION RUBRIC")
    output.append("="*70)
    output.append(f"\nRationale: {rubric['rationale']}\n")
    output.append("Criteria:")

    for criterion in rubric['criteria']:
        output.append(f"\n  • {criterion['name']} (Weight: {criterion['weight']:.2f})")
        output.append(f"    {criterion['description']}")

    return "\n".join(output)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Feature Discovery System - Generate features for ML models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python cli.py

  # With business problem
  python cli.py --problem "Predict customer churn in retail banking"

  # Save output to file
  python cli.py --problem "Detect fraud" --output results.json

  # Use OpenAI instead of Anthropic
  python cli.py --provider openai --model gpt-4

  # Increase iterations
  python cli.py --problem "Credit risk scoring" --iterations 5

  # Custom batch size for large feature sets
  python cli.py --problem "Credit risk" --batch-size 15

  # Enable parallel generation and evaluation (~2x speedup)
  python cli.py --problem "Credit risk" --parallel
        """
    )

    parser.add_argument(
        '--problem',
        type=str,
        help='Business problem description (if not provided, will prompt interactively)'
    )

    parser.add_argument(
        '--provider',
        type=str,
        choices=['anthropic', 'openai'],
        default='anthropic',
        help='LLM provider (default: anthropic)'
    )

    parser.add_argument(
        '--model',
        type=str,
        help='Model name (default: claude-sonnet-4-5-20250929 or gpt-4)'
    )

    parser.add_argument(
        '--iterations',
        type=int,
        default=3,
        help='Maximum iterations (default: 3)'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='LLM temperature (default: 0.7)'
    )

    parser.add_argument(
        '--score-threshold',
        type=float,
        default=7.0,
        help='Minimum score to keep features (default: 7.0)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Features per evaluation batch (default: 10)'
    )

    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Enable parallel generation and evaluation (~2x speedup)'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (JSON format)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    parser.add_argument(
        '--no-banner',
        action='store_true',
        help='Suppress banner'
    )

    args = parser.parse_args()

    # Configure logging
    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    elif args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    # Print banner
    if not args.no_banner:
        print_banner()

    # Get business problem
    if args.problem:
        business_problem = args.problem
    else:
        print("Enter your business problem (press Enter twice when done):")
        print("-" * 70)
        lines = []
        while True:
            line = input()
            if line:
                lines.append(line)
            else:
                break
        business_problem = "\n".join(lines)

        if not business_problem.strip():
            print("Error: No business problem provided")
            sys.exit(1)

    print(f"\n📋 Business Problem:")
    print(f"   {business_problem.strip()}\n")

    # Initialize workflow
    print(f"🔧 Initializing workflow...")
    print(f"   Provider: {args.provider}")
    print(f"   Model: {args.model or 'default'}")
    print(f"   Max iterations: {args.iterations}")
    print(f"   Score threshold: {args.score_threshold}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Parallel: {args.parallel}\n")

    try:
        workflow = FeatureDiscoveryWorkflow(
            llm_provider=args.provider,
            model_name=args.model,
            max_iterations=args.iterations,
            temperature=args.temperature,
            score_threshold=args.score_threshold,
            batch_size=args.batch_size,
            parallel=args.parallel
        )
    except Exception as e:
        print(f"❌ Error initializing workflow: {e}")
        sys.exit(1)

    # Run workflow
    print("🚀 Running feature discovery...")
    print("   This may take 1-3 minutes...\n")

    try:
        result = workflow.run(business_problem, verbose=args.verbose)
    except Exception as e:
        print(f"❌ Error running workflow: {e}")
        sys.exit(1)

    # Display results
    print("\n✅ Feature discovery complete!")
    print(f"   Total iterations: {result['total_iterations']}")

    print(format_rubric(result['rubric']))
    print(format_features(result['features'], result['evaluations']))

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Results saved to: {output_path}")

    print("\n" + "="*70)
    print("Thank you for using Feature Discovery System!")
    print("="*70)


if __name__ == "__main__":
    main()
