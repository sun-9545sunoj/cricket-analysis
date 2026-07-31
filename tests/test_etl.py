import os
import pytest
import duckdb
from src.etl import init_db, generate_sample_data, ingest_json_files, DEFAULT_DB_PATH, DEFAULT_RAW_DIR, get_connection

def test_data_ingestion_and_schema():
    """FR-1.1 Unit Test: Verifies ingestion of JSON match files and DuckDB schema."""
    # Ensure test sample data exists
    generate_sample_data(num_matches=10, raw_dir=DEFAULT_RAW_DIR)
    
    count = ingest_json_files(raw_dir=DEFAULT_RAW_DIR, db_path=DEFAULT_DB_PATH)
    assert count >= 10, f"Expected at least 10 matches in DB, found {count}"

    con = get_connection(DEFAULT_DB_PATH)
    matches_df = con.execute("SELECT * FROM matches LIMIT 5").df()
    deliveries_df = con.execute("SELECT * FROM deliveries LIMIT 5").df()
    con.close()

    assert not matches_df.empty, "Matches table should not be empty"
    assert not deliveries_df.empty, "Deliveries table should not be empty"

    required_match_cols = ["match_id", "venue", "team1", "team2", "winner", "target_runs"]
    for col in required_match_cols:
        assert col in matches_df.columns, f"Missing column {col} in matches table"

    required_deliv_cols = ["match_id", "over", "ball", "batter", "bowler", "total_runs", "is_wicket"]
    for col in required_deliv_cols:
        assert col in deliveries_df.columns, f"Missing column {col} in deliveries table"
