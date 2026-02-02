"""
Example: Feature Matching with Internal Feature Bank

This example shows how to:
1. Run the Feature Discovery workflow
2. Match generated features against an internal feature bank
3. Use different matching backends
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.workflow import FeatureDiscoveryWorkflow
from src.feature_matcher import create_matcher, FeatureMatchResult
from src.models.schemas import Feature


# Example internal feature bank (in production, load from your database/file)
INTERNAL_FEATURE_BANK = [
    {
        "name": "customer_age",
        "description": "Age of the customer in years",
        "table": "dim_customer",
        "column": "age"
    },
    {
        "name": "monthly_income",
        "description": "Average monthly income from all sources",
        "table": "fact_income",
        "column": "monthly_avg"
    },
    {
        "name": "account_balance",
        "description": "Current balance in primary checking account",
        "table": "fact_accounts",
        "column": "current_balance"
    },
    {
        "name": "credit_utilization_ratio",
        "description": "Ratio of used credit to total available credit limit",
        "table": "fact_credit",
        "column": "utilization_pct"
    },
    {
        "name": "num_transactions_30d",
        "description": "Number of transactions in the last 30 days",
        "table": "fact_transactions",
        "column": "txn_count_30d"
    },
    {
        "name": "avg_transaction_amount",
        "description": "Average transaction amount over last 90 days",
        "table": "fact_transactions",
        "column": "avg_txn_amt_90d"
    },
    {
        "name": "days_since_last_transaction",
        "description": "Days elapsed since the customer's last transaction",
        "table": "fact_transactions",
        "column": "days_since_last"
    },
    {
        "name": "loan_to_income_ratio",
        "description": "Total loan amount divided by annual income",
        "table": "fact_loans",
        "column": "lti_ratio"
    },
    {
        "name": "num_credit_inquiries_6m",
        "description": "Number of hard credit inquiries in last 6 months",
        "table": "fact_credit",
        "column": "hard_inquiries_6m"
    },
    {
        "name": "payment_history_score",
        "description": "Score based on on-time payment history (0-100)",
        "table": "fact_payments",
        "column": "payment_score"
    },
    {
        "name": "employment_tenure_months",
        "description": "Months at current employer",
        "table": "dim_employment",
        "column": "tenure_months"
    },
    {
        "name": "num_products_held",
        "description": "Total number of bank products the customer holds",
        "table": "dim_customer",
        "column": "product_count"
    },
]


def print_match_results(results: list[FeatureMatchResult], top_n: int = 3):
    """Pretty print match results."""
    print("\n" + "=" * 80)
    print("FEATURE MATCHING RESULTS")
    print("=" * 80)

    for result in results:
        print(f"\n Generated: {result.generated_feature}")
        print(f"   Desc: {result.generated_description[:60]}...")

        if result.matches:
            print("   Matches:")
            for i, match in enumerate(result.matches[:top_n]):
                print(f"     {i+1}. {match.internal_name} "
                      f"(score: {match.similarity_score:.3f})")
                if match.metadata.get('table'):
                    print(f"        -> {match.metadata['table']}.{match.metadata.get('column', '?')}")
        else:
            print("   No matches found")

    print("\n" + "=" * 80)


def example_with_workflow():
    """Run full workflow then match features."""
    print("\n" + "#" * 80)
    print("# EXAMPLE 1: Full Workflow + Matching")
    print("#" * 80)

    # Run feature discovery
    workflow = FeatureDiscoveryWorkflow(
        max_iterations=2,
        compact_mode=True,  # Faster for demo
        target_feature_count=10
    )

    business_problem = """
    Predict which customers are likely to default on their credit card payments
    in the next 90 days. Focus on behavioral and financial indicators.
    """

    print("\nRunning feature discovery...")
    result = workflow.run(business_problem)
    features = result["features"]

    print(f"\nGenerated {len(features)} features")

    # Match against internal bank
    print("\nMatching against internal feature bank...")
    matcher = create_matcher(backend="tfidf", top_k=3)  # TF-IDF for demo (no GPU)
    matcher.index(INTERNAL_FEATURE_BANK)

    match_results = matcher.match_all(features)
    print_match_results(match_results)

    # Summary statistics
    strong_matches = [r for r in match_results if r.matches and r.matches[0].similarity_score > 0.3]
    print(f"\nSummary: {len(strong_matches)}/{len(features)} features have potential internal matches")


def example_backend_comparison():
    """Compare different matching backends."""
    print("\n" + "#" * 80)
    print("# EXAMPLE 2: Compare Matching Backends")
    print("#" * 80)

    # Create some test features
    test_features = [
        Feature(
            name="avg_monthly_income",
            description="Average monthly income of the customer",
            data_type="structured",
            taxonomy="demographic",
            rationale="Income is predictive of repayment ability"
        ),
        Feature(
            name="credit_usage_percent",
            description="Percentage of available credit currently being used",
            data_type="structured",
            taxonomy="credit",
            rationale="High utilization indicates financial stress"
        ),
        Feature(
            name="transaction_frequency",
            description="How often the customer makes transactions",
            data_type="time_series",
            taxonomy="behavioral",
            rationale="Activity level indicates engagement"
        ),
    ]

    backends = ["fuzzy", "tfidf"]

    # Try to include vector if sentence-transformers is available
    try:
        import sentence_transformers
        backends.append("vector")
    except ImportError:
        print("\nNote: Install sentence-transformers for vector matching")

    for backend in backends:
        print(f"\n--- Backend: {backend.upper()} ---")
        matcher = create_matcher(backend=backend, top_k=3)
        matcher.index(INTERNAL_FEATURE_BANK)

        for feature in test_features:
            result = matcher.match_feature(feature)
            print(f"\n  {feature.name}:")
            for match in result.matches[:2]:
                print(f"    -> {match.internal_name}: {match.similarity_score:.3f}")


def example_standalone_matching():
    """Use matcher without running the full workflow."""
    print("\n" + "#" * 80)
    print("# EXAMPLE 3: Standalone Matching (No Workflow)")
    print("#" * 80)

    # You might have features from another source
    external_features = [
        Feature(
            name="customer_income",
            description="Total income",
            data_type="structured",
            taxonomy="",
            rationale=""
        ),
        Feature(
            name="account_age_days",
            description="How long the account has been open",
            data_type="structured",
            taxonomy="",
            rationale=""
        ),
    ]

    matcher = create_matcher(backend="fuzzy", top_k=5)
    matcher.index(INTERNAL_FEATURE_BANK)

    results = matcher.match_all(external_features)
    print_match_results(results, top_n=5)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Feature matching examples")
    parser.add_argument(
        "--example",
        choices=["workflow", "compare", "standalone", "all"],
        default="standalone",
        help="Which example to run"
    )

    args = parser.parse_args()

    if args.example in ("workflow", "all"):
        example_with_workflow()

    if args.example in ("compare", "all"):
        example_backend_comparison()

    if args.example in ("standalone", "all"):
        example_standalone_matching()
