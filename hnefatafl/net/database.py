from hnefatafl.net import DATABASE_FILE
from sqlite3 import Connection

import sqlite3


def connect() -> Connection:
    return sqlite3.connect(DATABASE_FILE)


def create(conn: Connection):
    conn.cursor().execute('CREATE TABLE users(username text PRIMARY KEY, password text)')
    conn.commit()


def save(conn: Connection, username: str, password: str):
    conn.cursor().execute(f'INSERT INTO users VALUES({username}, {password})')
    conn.commit()
