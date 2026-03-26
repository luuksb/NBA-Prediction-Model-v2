"""steps/ — Ordered preprocessing steps for the data pipeline.

Each step is a module with a single public function:
    run(df: pd.DataFrame) -> pd.DataFrame

Steps are pure functions: they receive a DataFrame and return a new one
without mutating the input. The pipeline runner in scripts/run_data_pipeline.py
chains them in order, persisting intermediate outputs to data/intermediate/.
"""
