#!/usr/bin/env python3
"""RAG Pipeline CLI fuer PDF Datenblaetter."""

import sys


def main():
    if len(sys.argv) < 2:
        print("Verwendung:")
        print("  python main.py init-db          - Datenbank initialisieren")
        print("  python main.py ingest <pfad> [-f|--force] - PDFs ingestieren")
        print("  python main.py chat             - Chat starten")
        sys.exit(1)

    command = sys.argv[1]

    if command == "init-db":
        from src.db import init_db
        init_db()

    elif command == "ingest":
        if len(sys.argv) < 3:
            print("Fehler: Pfad zum PDF-Ordner fehlt")
            print("Verwendung: python main.py ingest <pfad> [-f|--force]")
            sys.exit(1)

        from src.ingest import ingest_directory

        force = False
        if len(sys.argv) >= 4:
            if sys.argv[3] in ("-f", "--force"):
                force = True
            else:
                print(f"Unbekannte Option: {sys.argv[3]}")
                print("Verwendung: python main.py ingest <pfad> [-f|--force]")
                sys.exit(1)

        ingest_directory(sys.argv[2], force=force)

    elif command == "chat":
        from src.chat import chat_loop
        chat_loop()

    else:
        print(f"Unbekannter Befehl: {command}")
        print("Verfuegbare Befehle: init-db, ingest, chat")
        sys.exit(1)


if __name__ == "__main__":
    main()
