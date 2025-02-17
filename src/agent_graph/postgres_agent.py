import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

class PostgresAgent:
    def __init__(self):
        self.connection = psycopg2.connect(
            host=os.getenv("POSTGRES_DB_HOST"),
            port=os.getenv("POSTGRES_DB_PORT"),
            database=os.getenv("POSTGRES_DB_NAME"),
            user=os.getenv("POSTGRES_DB_USER"),
            password=os.getenv("POSTGRES_DB_PASSWORD")
        )
        self.cursor = self.connection.cursor()

    def execute_query(self, query):
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def close_connection(self):
        self.cursor.close()
        self.connection.close()