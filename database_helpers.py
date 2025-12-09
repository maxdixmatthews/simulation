import time
import psycopg2
import psycopg2.extras
import sqlalchemy as sa

def get_next_job(conn):
    """
    Atomically grab one pending job and mark it as running.
    Returns a dict with job fields, or None if no jobs.
    """
    with conn:  # opens a transaction
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT *
                FROM nd_job_queue
                WHERE status = 'pending' and ROUND(train_test_ratio::numeric, 2) = 0.2 and seed in (41, 42) and models in ('lr', 'lda') and dataset LIKE '%hospital%'
                ORDER BY created_at DESC
                FOR UPDATE SKIP LOCKED
                LIMIT 1
                """
            )
            job = cur.fetchone()
            if job is None:
                return None

            cur.execute(
                """
                UPDATE nd_job_queue
                SET status = 'running',
                    started_at = now()
                WHERE id = %s
                """,
                (job["id"],),
            )

    return dict(job)

def mark_job_done(conn, id: str):
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE nd_job_queue
                SET status = 'done',
                    finished_at = now()
                WHERE id = %s
                """,
                (id,),
            )


def mark_job_failed(conn, id: str, err: str):
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE nd_job_queue
                SET status = 'failed',
                    finished_at = now(),
                    last_error = %s
                WHERE id = %s
                """,
                (err[:1000], id),
            )