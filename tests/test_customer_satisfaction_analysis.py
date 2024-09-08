import pytest
import pandas as pd
from customer_satisfaction_analysis import (
    build_regression_model,
    run_kmeans_on_scores,
    aggregate_scores_per_cluster,
    assign_engagement_experience_scores
)

@pytest.fixture
def sample_data():
    # Create a sample DataFrame for testing
    data = {
        'MSISDN/Number': ['123', '456', '789'],
        'Engagement Score': [0.2, 0.5, 0.1],
        'Experience Score': [0.3, 0.6, 0.2],
        'Satisfaction Score': [0.25, 0.55, 0.15]
    }
    return pd.DataFrame(data)

def test_build_regression_model(sample_data):
    model, rmse = build_regression_model(sample_data)
    assert model is not None
    assert rmse >= 0

def test_run_kmeans_on_scores(sample_data):
    df, kmeans = run_kmeans_on_scores(sample_data)
    assert 'Score Cluster' in df.columns
    assert len(set(df['Score Cluster'])) == 2  # k=2

def test_aggregate_scores_per_cluster(sample_data):
    df, _ = run_kmeans_on_scores(sample_data)
    aggregated_df = aggregate_scores_per_cluster(df)
    assert 'Satisfaction Score' in aggregated_df.columns
    assert 'Experience Score' in aggregated_df.columns

def test_assign_engagement_experience_scores(sample_data):
    df = assign_engagement_experience_scores(sample_data, ['Engagement Score'], ['Experience Score'])
    assert 'Engagement Score' in df.columns
    assert 'Experience Score' in df.columns
