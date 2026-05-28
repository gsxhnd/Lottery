"""DuckDB 表结构定义"""

DRAWS_TABLE = "draws"

CREATE_DRAWS_TABLE = f"""
CREATE TABLE IF NOT EXISTS {DRAWS_TABLE} (
    issue VARCHAR PRIMARY KEY,
    draw_date DATE NOT NULL,
    red1 SMALLINT NOT NULL,
    red2 SMALLINT NOT NULL,
    red3 SMALLINT NOT NULL,
    red4 SMALLINT NOT NULL,
    red5 SMALLINT NOT NULL,
    red6 SMALLINT NOT NULL,
    blue SMALLINT NOT NULL,
    synced_at TIMESTAMP NOT NULL DEFAULT current_timestamp
)
"""

CREATE_DRAWS_DATE_INDEX = f"""
CREATE INDEX IF NOT EXISTS idx_{DRAWS_TABLE}_draw_date
ON {DRAWS_TABLE} (draw_date)
"""

SELECT_ALL_RECORDS = f"""
SELECT
    issue,
    CAST(draw_date AS VARCHAR) AS draw_date,
    red1, red2, red3, red4, red5, red6,
    blue
FROM {DRAWS_TABLE}
ORDER BY issue
"""

INSERT_DRAW = f"""
INSERT INTO {DRAWS_TABLE} (
    issue, draw_date, red1, red2, red3, red4, red5, red6, blue
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""
