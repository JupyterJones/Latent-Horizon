#!/usr/bin/env python3
import os
import re
import sys
import sqlite3
import requests

DB_FILE = "transcripts.db"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
CHAT_MODEL = "granite4.1:3b"
TOP_RESULTS = 4
MAX_HISTORY_MESSAGES = 12

SYSTEM_PROMPT = """You are a personal AI assistant with access to the user's historical conversation archive.
The historical archive is supplied as retrieved context from SQLite full-text search.
Use the retrieved historical conversations as evidence when they are relevant to the user's question.
Do not claim that something came from the historical archive unless the retrieved context actually supports it.
If the historical context does not contain the answer, say that the archive does not provide enough information rather than inventing a historical answer.
You should maintain continuity with the current conversation.
The user may ask follow-up questions referring to something you just discussed. Resolve those references using the current conversation first, and use the historical archive when appropriate.
Be concise but technically useful.
"""


def build_fts_query(raw_query: str) -> str:
    query = raw_query.strip()
    if not query:
        return ""

    if query.startswith('"') and query.endswith('"') and len(query) > 1:
        clean = query[1:-1].replace('"', '""')
        return f'"{clean}"'

    tokens = re.findall(r'[\w\-]+|\*', query)
    if not tokens:
        escaped = query.replace('"', '""')
        return f'"{escaped}"'

    fts_terms = []
    for token in tokens:
        if token.endswith('*') and len(token) > 1:
            clean_term = token[:-1].replace('"', '""')
            fts_terms.append(f'"{clean_term}"*')
        elif token != '*':
            clean_term = token.replace('"', '""')
            fts_terms.append(f'"{clean_term}"')

    # For natural language questions, use OR or a mix so it finds the most relevant matches
    return " OR ".join(fts_terms) if fts_terms else ""


def search_transcript(conn, question, limit=TOP_RESULTS):
    fts_query = build_fts_query(question)
    if not fts_query:
        return []

    cur = conn.cursor()
    sql = """
    SELECT c.id, c.conv_number, c.local_chunk, c.title, c.content,
           bm25(chunks_fts) AS rank
    FROM chunks_fts
    JOIN chunks c ON c.id = chunks_fts.rowid
    WHERE chunks_fts MATCH ?
    ORDER BY rank
    LIMIT ?
    """

    try:
        cur.execute(sql, (fts_query, limit))
        return cur.fetchall()
    except sqlite3.OperationalError:
        # Fallback to simple quoted query
        escaped = question.replace('"', '""')
        try:
            cur.execute(sql, (f'"{escaped}"', limit))
            return cur.fetchall()
        except sqlite3.OperationalError:
            return []


def build_history_text(results):
    if not results:
        return "No relevant historical conversation was found."

    sections = []
    for row in results:
        cid, conv_num, local_chunk, title, content, rank = row
        sections.append(
            f"""--- Historical Conversation {conv_num}, Part {local_chunk + 1} (Chunk #{cid}: {title}) ---
{content}
--- End Historical Conversation ---"""
        )
    return "\n\n".join(sections)


def ask_ollama(messages):
    response = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": CHAT_MODEL,
            "messages": messages,
            "stream": True
        },
        stream=True,
        timeout=800
    )
    response.raise_for_status()

    answer = ""
    for line in response.iter_lines():
        if not line:
            continue
        data = requests.models.complexjson.loads(line.decode("utf-8"))
        message = data.get("message", {})
        content = message.get("content", "")
        if content:
            print(content, end="", flush=True)
            answer += content
        if data.get("done"):
            break

    print()
    return answer


def main():
    if not os.path.exists(DB_FILE):
        print(f"Database not found: {DB_FILE}")
        print("Run 'python3 mkdown2sqlite.py' first to build the index.")
        return

    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM chunks")
    total_chunks = cur.fetchone()[0]

    print()
    print("=" * 70)
    print("TRANSCRIPT CHAT (Powered by SQLite3 FTS5)")
    print("=" * 70)
    print(f"Model:                {CHAT_MODEL}")
    print(f"Database:             {DB_FILE}")
    print(f"Indexed Chunks:       {total_chunks:,}")
    print(f"Retrieval Speed:      ~10ms (No GPU / CPU embedding required!)")
    print()
    print("Type 'quit' or 'exit' to stop.")
    print("Type 'clear' to clear current conversation memory.")
    print("=" * 70)
    print()

    conversation_history = []

    try:
        while True:
            try:
                question = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                break

            if not question:
                continue

            if question.lower() in {"quit", "exit"}:
                break

            if question.lower() == "clear":
                conversation_history = []
                print("Current conversation memory cleared.\n")
                continue

            print("\nSearching transcripts (SQLite FTS5)...", flush=True)
            results = search_transcript(conn, question, limit=TOP_RESULTS)

            matched_chunks = [f"#{r[0]} (Conv {r[1]})" for r in results]
            if matched_chunks:
                print(f"Found {len(results)} relevant chunks: {', '.join(matched_chunks)}")
            else:
                print("No direct keyword match found in archive.")

            historical_context = build_history_text(results)

            user_message = f"""Historical conversation context:
{historical_context}

Current conversation:
{chr(10).join(
    f"{message['role']}: {message['content']}"
    for message in conversation_history[-MAX_HISTORY_MESSAGES:]
)}

User's new question:
{question}
"""

            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                }
            ]
            messages.extend(conversation_history[-MAX_HISTORY_MESSAGES:])
            messages.append(
                {
                    "role": "user",
                    "content": user_message
                }
            )

            print(f"\n{CHAT_MODEL}:")
            try:
                answer = ask_ollama(messages)
            except Exception as e:
                print(f"\nOllama error: {e}\n")
                continue

            conversation_history.append({"role": "user", "content": question})
            conversation_history.append({"role": "assistant", "content": answer})

            if len(conversation_history) > MAX_HISTORY_MESSAGES:
                conversation_history = conversation_history[-MAX_HISTORY_MESSAGES:]
            print()

    finally:
        conn.close()


if __name__ == "__main__":
    main()
