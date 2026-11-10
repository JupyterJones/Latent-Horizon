#!/usr/bin/env python3
import os
import re
import time
import sqlite3
from icecream import ic

TRANSCRIPT_FILE = "all_transcript_full.md"
DB_FILE = "transcripts.db"

CHUNK_SIZE = 4000
CHUNK_OVERLAP = 400


def read_transcript(filepath=TRANSCRIPT_FILE):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Transcript not found: {filepath}")

    print(f"Reading {filepath}...")
    t0 = time.time()
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    print(f"Read {len(text):,} characters ({len(text.encode('utf-8')) / (1024 * 1024):.1f} MB) in {time.time() - t0:.2f}s")
    return text


def split_conversations(text):
    pattern = r"(?=^# )"
    conversations = re.split(pattern, text, flags=re.MULTILINE)
    conversations = [c.strip() for c in conversations if c.strip()]
    ic(len(conversations))
    return conversations


def extract_title(text, max_len=100):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines:
        if line.startswith("<USER_REQUEST>"):
            continue
        cleaned = re.sub(r"^#+\s*", "", line)
        if cleaned:
            return cleaned[:max_len]
    return "Untitled Conversation"


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            paragraph_break = text.rfind("\n\n", start, end)
            if paragraph_break > start + (chunk_size // 2):
                end = paragraph_break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def build_sqlite_db(transcript_text, db_path=DB_FILE):
    print("Splitting conversations and creating chunks...")
    conversations = split_conversations(transcript_text)

    # Temporary database replacement or clean start
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Fast PRAGMA settings for bulk loading
    cur.execute("PRAGMA journal_mode = WAL;")
    cur.execute("PRAGMA synchronous = NORMAL;")

    cur.execute("""
    CREATE TABLE chunks (
        id INTEGER PRIMARY KEY,
        conv_number INTEGER NOT NULL,
        local_chunk INTEGER NOT NULL,
        title TEXT,
        content TEXT NOT NULL,
        char_count INTEGER NOT NULL
    );
    """)

    cur.execute("""
    CREATE INDEX idx_chunks_conv ON chunks(conv_number);
    """)

    cur.execute("""
    CREATE VIRTUAL TABLE chunks_fts USING fts5(
        title,
        content,
        content='chunks',
        content_rowid='id',
        tokenize='porter unicode61'
    );
    """)

    chunks_data = []
    fts_data = []
    chunk_id = 1

    t0 = time.time()
    for conv_num, conv in enumerate(conversations):
        conv_title = extract_title(conv)
        sub_chunks = chunk_text(conv)

        for local_num, chunk in enumerate(sub_chunks):
            # Chunk title: conv title plus chunk indicator if multiple
            title = f"{conv_title} [part {local_num + 1}]" if len(sub_chunks) > 1 else conv_title
            char_count = len(chunk)

            chunks_data.append((chunk_id, conv_num, local_num, title, chunk, char_count))
            fts_data.append((chunk_id, title, chunk))
            chunk_id += 1

    print(f"Prepared {len(chunks_data):,} chunks from {len(conversations):,} conversations in {time.time() - t0:.2f}s")

    print(f"Indexing into {db_path}...")
    t_insert = time.time()
    cur.executemany(
        "INSERT INTO chunks (id, conv_number, local_chunk, title, content, char_count) VALUES (?, ?, ?, ?, ?, ?)",
        chunks_data
    )
    cur.executemany(
        "INSERT INTO chunks_fts (rowid, title, content) VALUES (?, ?, ?)",
        fts_data
    )
    conn.commit()

    # Optimize FTS index
    cur.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize');")
    conn.commit()
    conn.close()

    db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
    print(f"FTS5 indexing completed in {time.time() - t_insert:.2f}s!")
    print()
    print("=" * 60)
    print("DATABASE READY")
    print(f"File:         {db_path}")
    print(f"Size:         {db_size_mb:.2f} MB")
    print(f"Conversations: {len(conversations):,}")
    print(f"Total Chunks:  {len(chunks_data):,}")
    print("=" * 60)


def main():
    start_time = time.time()
    text = read_transcript()
    build_sqlite_db(text)
    print(f"\nTotal elapsed time: {time.time() - start_time:.2f} seconds\n")


if __name__ == "__main__":
    main()
