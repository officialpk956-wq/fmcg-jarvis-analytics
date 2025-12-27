import pandas as pd

def sales_summary(conn, year: int = None):
    query = """
        SELECT SUM(units_sold) AS total_units
        FROM sales
    """
    if year:
        query += f" WHERE strftime('%Y', date) = '{year}'"

    return pd.read_sql(query, conn)["total_units"].iloc[0]


def top_category(conn):
    query = """
        SELECT category, SUM(units_sold) AS total_units
        FROM sales
        GROUP BY category
        ORDER BY total_units DESC
        LIMIT 1
    """
    return pd.read_sql(query, conn).iloc[0].to_dict()


def region_performance(conn):
    query = """
        SELECT region, SUM(units_sold) AS total_units
        FROM sales
        GROUP BY region
        ORDER BY total_units DESC
        LIMIT 1
    """
    return pd.read_sql(query, conn).iloc[0].to_dict()


def promo_effect(conn):
    query = """
        SELECT promotion_flag, AVG(units_sold) AS avg_units
        FROM sales
        GROUP BY promotion_flag
    """
    df = pd.read_sql(query, conn)
    return {
        "no_promo": df[df["promotion_flag"] == 0]["avg_units"].iloc[0],
        "promo": df[df["promotion_flag"] == 1]["avg_units"].iloc[0],
    }
