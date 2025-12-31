# Prompt: KI-Schach-App (Self-Play / AlphaZero-lite) mit 3D-Board, Live-Training, NVIDIA-GPU, GitOps im Container, One-Command-Run

Du bist ein Senior-Engineering-Team (Tech Lead, ML Engineer, Fullstack Engineer, DevOps) und baust ein **produktionsreifes** Open-Source-Projekt (MIT License), das als Web-App läuft.  
Ziel ist eine KI-Schach-App, die sich **selbst Schach beibringt (Self-Play / AlphaZero-lite)**, deren Lernfortschritt **live sichtbar** ist, und gegen die ich anschließend spielen kann.  
Die App läuft auf einem **selbstgehosteten Ubuntu-Server** und nutzt eine **NVIDIA GPU (CUDA)** für Training/Inference.

**WICHTIG:** Alles (inkl. Git-Repo-Überwachung, Auto-Pull, Auto-Deploy, Auto-Commit/Push, Changelog/README Updates) muss **im Container** laufen.  
**One-Command-Run:** `docker compose up -d` startet **alles** vollständig lauffähig.

---

## 1) Zielbild / User Experience (Vibe)

### Kernerlebnis
- Modern, ruhig, premium UI (clean, responsive, schnelle Ladezeiten).
- **Sehr schön gerendertes Schachbrett mit Figuren** (bevorzugt 3D).
- Zusätzlich ein **Trainingsmodus**, in dem ich live sehe, wie die KI trainiert:
  - Live-Metriken: Policy-Loss, Value-Loss, Winrate, Elo-Schätzung, Throughput
  - laufende Self-Play-Partien (Liste + Mini-Board)
  - “Watch a game” Replay (Animation, Züge, Ergebnis)
  - aktuelles Modell/Checkpoint, “best model”, Promotion-Historie

### Modi
1) **Play:** Ich spiele gegen die KI (Schwierigkeitsstufen).
2) **Train Live:** Live-Dashboard (Charts + Spiele + Promoted Models).
3) **Replay/Explain:** Partien abspielen, optional Top-3 Züge + kurze Begründung.

---

## 2) Stack & Architektur

### Services (Docker Compose)
1) `frontend` (React + Three.js)
2) `backend` (FastAPI + REST + WebSockets)
3) `trainer` (PyTorch CUDA: Self-Play + Training + Eval + Checkpoints)
4) `gitops` (**im Container!**): Repo-Watcher + Auto-Pull + Auto-Redeploy + Auto-Commit/Push + Changelog/README Update
5) optional: `db` (Postgres) oder Start mit SQLite

### Grundprinzip
- `trainer` nutzt GPU, führt Self-Play, Training, Evaluation aus und schreibt Metriken + Spiele.
- `backend` streamt Metriken und Self-Play-Spiele via WebSocket an `frontend`.
- `frontend` zeigt 3D-Brett + Training Dashboard.
- `gitops` überwacht das Git-Repo und steuert Redeploy/Build sowie automatische Commits.

---

## 3) Backend Anforderungen (FastAPI)

- python-chess für Regeln/Legal Moves.
- WebSockets:
  - `/ws/training` streamt Events in Echtzeit.
- REST APIs:
  - `/api/game/new`, `/api/game/state`, `/api/game/move`, `/api/game/legal`
  - `/api/models`, `/api/models/active`, `/api/models/set`
  - `/api/replays`, `/api/replays/{id}`
  - `/api/changelog` (komprimierte Changelog-Einträge fürs UI)
- Konfiguration über `.env` (niemals committen).

---

## 4) Trainer Anforderungen (PyTorch + CUDA)

- Policy+Value Netz (ResNet-lite).
- Self-Play via MCTS:
  - Inference auf GPU, Tree auf CPU
  - Batch-Inference in MCTS
  - AMP (mixed precision)
- Checkpoints:
  - versioniert `models/model_000XX.pt`
  - “best model” markieren
- Evaluation:
  - neues Modell vs best/previous (W/D/L, Elo-Schätzung)
  - Promotion wenn besser
- Trainer schreibt Live-Metriken/Events, die `backend` streamen kann.

---

## 5) Frontend Anforderungen (React + Three.js)

- 3D Brett + Figuren (glTF), PBR, saubere Kamera, Soft Shadows.
- Moves animiert (lift → slide → drop), Captures dezent.
- UI:
  - Play-Ansicht (Schwierigkeitsgrad, Start, Reset)
  - Training-Ansicht (Charts + Live-Spiele + Replay)
  - Replay-Ansicht (Zugliste, Animation, Ergebnis)
- Mobile-first, responsiv.

---

## 6) GitOps im Container (Kernanforderung)

### Grundsatz
**Kein systemd/Host-Setup.** Alles läuft containerisiert und wird durch `docker compose up -d` gestartet.

### A) Repo-Überwachung → Auto Pull → Auto Redeploy
Implementiere einen `gitops`-Service, der:
1) periodisch `git fetch` macht (z.B. alle 10–30 Sekunden) ODER optional Webhook (wenn vorhanden)
2) bei neuen Commits:
   - `git pull`
   - `docker compose build` (falls nötig)
   - `docker compose up -d` (rolling, ohne Downtime soweit möglich)
   - Healthcheck auf `backend` und `frontend`
   - bei Fehler: automatischer Rollback auf vorheriges Release/Image (falls möglich)

### B) Automatischer Commit & Push bei lokalen Änderungen
Wenn innerhalb des Workdir Änderungen entstehen (z.B. durch Generator/Agent, Auto-Changelog, Auto-README):
1) `git status` prüfen
2) `git add -A` (nur erlaubte Dateien)
3) Commit-Message automatisch generieren, kurz & konventionell (`feat:`, `fix:`, `chore:`)
4) `git push`

**Sicherheitsregeln:**
- `.env` und Secrets nie committen.
- `.gitignore` konsequent.
- Credentials (GitHub Token/SSH Key) ausschließlich über Docker Secrets oder `.env` injection; niemals im Repo.

### C) CHANGELOG & README Updates
Bei jeder Änderung:
- `CHANGELOG.md` kompakt aktualisieren
- `README.md` (Setup, GPU, Compose, URLs, Troubleshooting)
- `backend` stellt `/api/changelog` bereit, `frontend` zeigt es an

---

## 7) Wie GitOps containerisiert wird (klar umsetzen)

### Anforderungen an `gitops` Container
- Muss Zugriff auf:
  - das Repo (als Volume)
  - den Docker Socket (`/var/run/docker.sock`) **oder** Docker Remote API, um Compose Commands auszuführen
- Muss eine robuste Loop haben (watch/poll), Logs schreiben und Fehler sauber handhaben
- Muss “Single Source of Truth” sein: alles was deployed wird kommt aus Git

**Hinweis:** Wenn Docker Socket genutzt wird, dokumentiere Security-Implikationen und biete optional Remote API Alternative.

---

## 8) Volumes / Projektlayout (damit Watcher + App konsistent sind)

- Repo wird als Volume in alle Services gemountet (read-only wo möglich).
- Modelle, Metriken, PGNs werden in persistenten Volumes gespeichert.
- `gitops` darf nur bestimmte Pfade committen (z.B. `CHANGELOG.md`, `README.md`, `docs/`, code changes) – aber niemals `models/` und nie große Binärdaten (konfigurierbar).

---

## 9) Qualität

- Produktionsreife Struktur, modulare Packages.
- Tests: chess move encoding, API smoke tests.
- Observability: strukturierte Logs, einfache Metriken.
- Performance: flüssige UI, effizientes GPU Training (AMP + Batch).
- Dokumentation vollständig.

---

## 10) Deliverables (alles muss enthalten sein)

1) Vollständige Repo-Struktur mit allen Dateien.
2) `docker-compose.yml` inkl. GPU, Healthchecks, Volumes, Networks.
3) `frontend` kompletter Code (3D Board + Dashboard + Replay).
4) `backend` kompletter Code (REST + WS + changelog endpoint).
5) `trainer` kompletter Code (self-play + training + eval + checkpoints).
6) `gitops` kompletter Code:
   - Repo-Watcher (poll/webhook optional)
   - Auto Pull / Auto Deploy (Compose orchestration)
   - Auto Commit/Push (inkl. Sicherheitsfilter)
   - Auto Update CHANGELOG/README
7) `.env.example` + klare Setup Anleitung für:
   - NVIDIA Treiber / Container Toolkit
   - GitHub Auth (Token/SSH via secrets)
   - Start/Stop/Logs
8) `CHANGELOG.md` & `README.md` initial, Mechanismus zum weiteren Update.

---

## 11) Output Format (sehr wichtig)

- Gib die gesamte Lösung als **Dateistruktur** aus.
- Jede Datei in einem separaten Codeblock, mit Dateipfad als Überschrift.
- Alles muss lauffähig sein, ohne “später ausfüllen”.
- Für große Assets (glTF Figuren): nutze minimal lauffähige Defaults oder kleine freie Assets + dokumentierte Download-Schritte.

---

## Jetzt starten

Erstelle zuerst ein lauffähiges End-to-End MVP:
- 3D Board + Play vs Dummy AI
- Training Dashboard mit Dummy Events via WebSocket
- `gitops` Container der Pull/Deploy Loop zeigt (auch wenn initial nur “noop”)
Dann erweitern:
- echtes AlphaZero-lite Training (CUDA)
- Evaluation + Promotion
- Performanceoptimierungen (AMP + batch inference)
- vollständige GitOps Automationen (commit/push + changelog/readme)
