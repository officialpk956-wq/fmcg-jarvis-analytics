import pandas as pd
import logging
import hashlib
from dataclasses import dataclass
from typing import Optional, Any, Dict
from datetime import datetime

from sqlalchemy.engine import Engine

"""
SQL query functions for FMCG Jarvis.

All functions accept a SQLAlchemy Engine and use explicit connection
context managers for each query. This ensures proper connection lifecycle
management with SQLAlchemy 2.0+ and prevents stale connection issues
on Streamlit Cloud.

Design Decision: Engine vs Connection
-------------------------------------
We pass Engine (not Connection) because:
1. Engine manages a connection pool - efficient for multiple queries
2. Connections should be short-lived (within `with engine.connect()` blocks)
3. Engine survives Streamlit reruns without stale connection issues
4. Each query gets a fresh connection from the pool

Key pattern:
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

This approach:
- Creates a fresh connection for each query
- Automatically returns connection to pool after use
- Prevents connection state issues across Streamlit reruns
- Works reliably on Streamlit Cloud
"""

# Configure module logger
logger = logging.getLogger(__name__)


# -------------------------
# SQL ERROR HANDLING
# -------------------------
@dataclass
class SQLErrorContext:
    """
    Structured error context for SQL failures.
    Contains safe-to-display info without leaking sensitive data.
    """
    error_id: str           # Unique ID for log correlation
    error_type: str         # Exception class name
    operation: str          # What operation was attempted
    timestamp: str          # When error occurred
    user_message: str       # Safe message for UI
    technical_hint: str     # Safe hint for debugging
    
    def __str__(self):
        return f"[{self.error_id}] {self.user_message}"


class SQLExecutionError(Exception):
    """
    Exception raised when SQL execution fails.
    Wraps the original exception with safe context.
    """
    def __init__(self, context: SQLErrorContext, original: Exception):
        self.context = context
        self.original = original
        super().__init__(str(context))


def _generate_error_id() -> str:
    """Generate a short unique error ID for log correlation."""
    timestamp = datetime.now().isoformat()
    return hashlib.md5(timestamp.encode()).hexdigest()[:8].upper()


def _classify_sql_error(exc: Exception) -> tuple[str, str]:
    """
    Classify a SQL exception into user-friendly message and technical hint.
    Does NOT expose the actual SQL query or sensitive details.
    
    Returns:
        (user_message, technical_hint)
    """
    exc_name = type(exc).__name__
    exc_str = str(exc).lower()
    
    # Connection errors
    if "connection" in exc_str or "connect" in exc_str:
        return (
            "Database connection failed.",
            "Check database file exists and is accessible."
        )
    
    # Table/column not found
    if "no such table" in exc_str:
        return (
            "Required database table not found.",
            "Database schema may be incomplete. Re-run data pipeline."
        )
    if "no such column" in exc_str:
        return (
            "Required data column not found.",
            "Database schema may have changed. Check table structure."
        )
    
    # Syntax errors (shouldn't happen with static queries)
    if "syntax" in exc_str:
        return (
            "Internal query error.",
            "Please report this issue with error ID."
        )
    
    # Timeout
    if "timeout" in exc_str or "timed out" in exc_str:
        return (
            "Database query timed out.",
            "Query may be too complex or database is under heavy load."
        )
    
    # Lock/busy
    if "locked" in exc_str or "busy" in exc_str:
        return (
            "Database is temporarily busy.",
            "Another operation is in progress. Please retry."
        )
    
    # Pandas-specific
    if exc_name == "DatabaseError":
        return (
            "Database operation failed.",
            "Check database connectivity and permissions."
        )
    
    # Generic fallback
    return (
        "An unexpected database error occurred.",
        f"Error type: {exc_name}. Check logs with error ID for details."
    )


def execute_sql(
    query: str,
    engine: Engine,
    params: Optional[Dict[str, Any]] = None,
    operation: str = "query"
) -> pd.DataFrame:
    """
    Execute a SQL query with structured error handling.
    
    Args:
        query: SQL query string
        engine: SQLAlchemy Engine (NOT a Connection)
        params: Optional query parameters
        operation: Description of the operation (for error messages)
        
    Returns:
        pandas DataFrame with results
        
    Raises:
        SQLExecutionError: On any database error, with safe context
    """
    try:
        with engine.connect() as conn:
            return pd.read_sql(query, conn, params=params)
            
    except Exception as exc:
        error_id = _generate_error_id()
        user_msg, tech_hint = _classify_sql_error(exc)
        
        context = SQLErrorContext(
            error_id=error_id,
            error_type=type(exc).__name__,
            operation=operation,
            timestamp=datetime.now().isoformat(),
            user_message=user_msg,
            technical_hint=tech_hint,
        )
        
        # Log the FULL error for debugging (includes query in logs only)
        logger.error(
            f"SQL Error [{error_id}] during '{operation}': "
            f"{type(exc).__name__}: {exc}"
        )
        
        raise SQLExecutionError(context, exc) from exc


def _year_filter_params(year: int) -> dict:
    """
    Generate parameters for parameterized year filter queries.
    
    Returns dict with:
    - year_substr: '2024' for SUBSTR matching
    - year_dash: '2024-%' for ISO with dash
    - year_slash: '2024/%' for ISO with slash
    """
    return {
        "year_substr": str(year),
        "year_dash": f"{year}-%",
        "year_slash": f"{year}/%",
    }


# -------------------------
# SAFE DATAFRAME HELPERS
# -------------------------
def safe_get_scalar(df, column, default=None):
    """
    Safely extract a scalar value from a DataFrame.
    Returns default if DataFrame is empty or column is missing.
    """
    if df is None or df.empty:
        return default
    if column not in df.columns:
        return default
    value = df[column].iloc[0]
    return value if value is not None else default


def safe_get_row(df, as_dict=True):
    """
    Safely extract the first row from a DataFrame.
    Returns None if empty, or dict/Series based on as_dict flag.
    """
    if df is None or df.empty:
        return None
    row = df.iloc[0]
    return row.to_dict() if as_dict else row


def safe_get_filtered_scalar(df, filter_col, filter_val, value_col, default=None):
    """
    Safely extract a scalar from a filtered DataFrame.
    Example: safe_get_filtered_scalar(df, 'promotion_flag', 0, 'avg_units', 0.0)
    """
    if df is None or df.empty:
        return default
    filtered = df[df[filter_col] == filter_val]
    if filtered.empty:
        return default
    value = filtered[value_col].iloc[0]
    return value if value is not None else default


# -------------------------
# SQL QUERY FUNCTIONS
# -------------------------
def sales_summary(engine: Engine, year: Optional[int] = None) -> int:
    """
    Get total units sold, optionally filtered by year.
    Uses robust TEXT date matching via SUBSTR + LIKE fallbacks.
    
    Args:
        engine: SQLAlchemy Engine
        year: Optional 4-digit year to filter on
        
    Returns:
        Total units sold (0 if no data)
        
    Raises:
        SQLExecutionError: On database errors with safe context
    """
    if year:
        # SUBSTR(date, 1, 4) extracts first 4 chars - works for:
        # '2024-01-01', '2024/01/01', '20240101', '2024-01-01 10:30:00'
        # LIKE patterns catch edge cases with different separators
        query = """
            SELECT COALESCE(SUM(units_sold), 0) AS total_units
            FROM sales
            WHERE SUBSTR(date, 1, 4) = :year_substr
               OR date LIKE :year_dash
               OR date LIKE :year_slash
        """
        df = execute_sql(query, engine, params=_year_filter_params(year),
                        operation=f"sales summary for {year}")
    else:
        query = """
            SELECT COALESCE(SUM(units_sold), 0) AS total_units
            FROM sales
        """
        df = execute_sql(query, engine, operation="total sales summary")
    
    return safe_get_scalar(df, "total_units", default=0)


def top_category(engine: Engine) -> Dict[str, Any]:
    """
    Get the best performing category by units sold.
    
    Args:
        engine: SQLAlchemy Engine
        
    Returns:
        Dict with 'category', 'total_units', and '_empty' flag
    
    Raises:
        SQLExecutionError: On database errors with safe context
    """
    query = """
        SELECT 
            COALESCE(category, 'Unknown') AS category, 
            COALESCE(SUM(units_sold), 0) AS total_units
        FROM sales
        WHERE category IS NOT NULL
        GROUP BY category
        ORDER BY total_units DESC
        LIMIT 1
    """
    df = execute_sql(query, engine, operation="top category lookup")
    
    result = safe_get_row(df, as_dict=True)
    if result is None:
        return {"category": "No data", "total_units": 0, "_empty": True}
    
    return result


def region_performance(engine: Engine) -> Dict[str, Any]:
    """
    Get the best performing region by units sold.
    
    Args:
        engine: SQLAlchemy Engine
        
    Returns:
        Dict with 'region', 'total_units', and '_empty' flag
    
    Raises:
        SQLExecutionError: On database errors with safe context
    """
    query = """
        SELECT 
            COALESCE(region, 'Unknown') AS region, 
            COALESCE(SUM(units_sold), 0) AS total_units
        FROM sales
        WHERE region IS NOT NULL
        GROUP BY region
        ORDER BY total_units DESC
        LIMIT 1
    """
    df = execute_sql(query, engine, operation="region performance lookup")
    
    result = safe_get_row(df, as_dict=True)
    if result is None:
        return {"region": "No data", "total_units": 0, "_empty": True}
    
    return result


def promo_effect(engine: Engine) -> Dict[str, Any]:
    """
    Get average sales by promotion flag.
    
    Args:
        engine: SQLAlchemy Engine
        
    Returns:
        Dict with 'promo', 'no_promo' averages, and '_empty' flag
    
    Raises:
        SQLExecutionError: On database errors with safe context
    """
    query = """
        SELECT 
            promotion_flag, 
            COALESCE(AVG(units_sold), 0) AS avg_units
        FROM sales
        WHERE promotion_flag IN (0, 1)
        GROUP BY promotion_flag
    """
    df = execute_sql(query, engine, operation="promotion effect analysis")
    
    no_promo = safe_get_filtered_scalar(df, "promotion_flag", 0, "avg_units", default=0.0)
    promo = safe_get_filtered_scalar(df, "promotion_flag", 1, "avg_units", default=0.0)
    
    return {
        "no_promo": float(no_promo),
        "promo": float(promo),
        "_empty": df.empty,
    }
