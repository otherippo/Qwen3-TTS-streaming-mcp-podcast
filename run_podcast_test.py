#!/usr/bin/env python3
"""
3-voice podcast test against local TTS extended proxy.
Collects detailed timing data and monitors job to completion.
"""

import os
import requests
import time
import sys
from datetime import datetime

# Configuration
BASE_URL = os.getenv("BASE_URL", "http://localhost:8883")
SERVER_HOSTNAME = os.getenv("SERVER_HOSTNAME", "localhost")
TIMEOUT = 900  # seconds for overall script

PODCAST_TURNS = [
    {"speaker": "Anke", "text": "Willkommen zur heutigen Folge unseres kleinen Podcasts. Ich bin Anke und freue mich, dass ihr wieder dabei seid. Heute haben wir ein spannendes Thema."},
    {"speaker": "Gerd", "text": "Hallo zusammen, Gerd hier. Ja, das Thema ist wirklich interessant. Wir wollen heute darueber sprechen, wie kuenstliche Intelligenz unseren Alltag veraendert."},
    {"speaker": "MargritS", "text": "Guten Tag, ich bin Margrit. Ich finde, das ist ein wichtiges Thema. Viele Menschen haben Angst vor KI, aber ich sehe auch grosse Chancen."},
    {"speaker": "Anke", "text": "Das stimmt, Margrit. Die Chancen sind enorm. Denkt nur an die Medizin. KI kann Aertzte bei der Diagnose unterstuetzen und Leben retten."},
    {"speaker": "Gerd", "text": "Aber wir sollten auch die Risiken nicht ausser Acht lassen. Datenschutz ist ein grosses Thema. Wer kontrolliert eigentlich all diese Algorithmen?"},
    {"speaker": "MargritS", "text": "Da hast du recht, Gerd. Transparenz ist entscheidend. Wir brauchen Regeln, damit die Technologie zum Wohl der Menschheit eingesetzt wird."},
    {"speaker": "Anke", "text": "Ich glaube, Bildung spielt eine zentrale Rolle. Je besser die Menschen verstehen, wie KI funktioniert, desto selbstbewusster koennen sie damit umgehen."},
    {"speaker": "Gerd", "text": "Absolut. In Schulen und Universitaeten sollte Informatik und Ethik Hand in Hand gehen. Die naechste Generation muss darauf vorbereitet sein."},
    {"speaker": "MargritS", "text": "Und was meint ihr zur Kreativitaet? Koennen Maschinen wirklich Kunst und Musik erschaffen, die Emotionen beruehren?"},
    {"speaker": "Anke", "text": "Das ist eine philosophische Frage. Ich denke, KI kann Werkzeuge sein, die Kreative unterstuetzen. Aber die Seele kommt vom Menschen."},
    {"speaker": "Gerd", "text": "Interessanter Punkt. Vielleicht ist der Unterschied gar nicht so wichtig. Wenn das Ergebnis schoen ist, zaehlt doch die Erfahrung des Betrachters."},
    {"speaker": "MargritS", "text": "Hmmm, da bin ich mir nicht so sicher. Herkunft und Absicht matteren fuer mich. Ein Kunstwerk traegt die Geschichte seines Schoepfers in sich."},
    {"speaker": "Anke", "text": "Lasst uns zusammenfassen: KI bietet Chancen in Medizin, Bildung und Kreativitaet. Aber wir brauchen Ethik, Transparenz und Bildung."},
    {"speaker": "Gerd", "text": "Genau. Die Technologie an sich ist weder gut noch boese. Es kommt darauf an, wie wir sie nutzen. Verantwortung liegt bei uns allen."},
    {"speaker": "MargritS", "text": "Schoen gesagt. Ich hoffe, unsere Hoerer nehmen sich das zu Herzen. Der Dialog zwischen Mensch und Maschine ist erst am Anfang."},
    {"speaker": "Anke", "text": "Vielen Dank an Gerd und Margrit fuer diese tolle Diskussion. Und danke an euch fuers Zuhoeren. Bis zur naechsten Folge!"},
    {"speaker": "Gerd", "text": "Tschuess und auf Wiederhoeren. Bleibt neugierig und kritisch. Die Zukunft gestalten wir gemeinsam."},
    {"speaker": "MargritS", "text": "Auf Wiedersehen, liebe Hoerer. Denkt daran: Technologie ist ein Werkzeug. Die Vision kommt von uns Menschen. Macht es gut!"},
]

def submit_podcast():
    """Submit the podcast job."""
    payload = {
        "turns": PODCAST_TURNS,
        "language": "German",
        "model": "1.7B",
        "no_drift": True
    }
    resp = requests.post(f"{BASE_URL}/api/podcast", json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    job_id = data.get("id") or data.get("job_id")
    print(f"[SUBMIT] Job ID: {job_id}")
    print(f"[SUBMIT] Response keys: {list(data.keys())}")
    if not job_id:
        raise RuntimeError("No job ID returned from API")
    return job_id

def get_job_status(job_id):
    """Get current job status."""
    resp = requests.get(f"{BASE_URL}/api/jobs/{job_id}", timeout=30)
    resp.raise_for_status()
    return resp.json()

def retry_chunk(job_id, chunk_idx):
    """Retry a failed chunk."""
    resp = requests.post(f"{BASE_URL}/api/jobs/{job_id}/retry/{chunk_idx}", timeout=30)
    resp.raise_for_status()
    return resp.json()

def format_duration(seconds):
    """Format seconds to human readable."""
    return f"{seconds:.2f}s"

def main():
    start_time = time.time()
    script_deadline = start_time + TIMEOUT
    
    # Submit
    print("="*70)
    print("PODCAST TTS BENCHMARK")
    print("="*70)
    job_id = submit_podcast()
    
    # Tracking structures
    chunk_stats = {}  # idx -> {first_seen, completed_at, failed_count, last_error, voice, chars}
    retries_log = []  # list of {chunk_idx, attempt, result}
    eta_snapshots = []  # list of {elapsed, eta, status}
    
    # Pre-populate chunk info
    for idx, turn in enumerate(PODCAST_TURNS):
        chunk_stats[idx] = {
            "voice": turn["speaker"],
            "chars": len(turn["text"]),
            "first_seen": None,
            "completed_at": None,
            "elapsed": None,
            "failed_count": 0,
            "last_error": None,
            "final_status": "unknown"
        }
    
    initial_eta = None
    last_status_str = ""
    
    print("\n[POLLING] Starting polling loop every 3 seconds...")
    print("-"*70)
    
    try:
        while time.time() < script_deadline:
            elapsed_total = time.time() - start_time
            
            try:
                status = get_job_status(job_id)
            except Exception as e:
                print(f"[WARN] Poll error at {format_duration(elapsed_total)}: {e}")
                time.sleep(3)
                continue
            
            current_eta = status.get("eta_seconds")
            state = status.get("status", "unknown")
            chunks = status.get("chunks", [])
            failed_chunks = [c for c in chunks if c.get("status") == "failed"]
            
            # Record ETA snapshot on status changes or every ~15s
            status_str = f"{state}|done:{status.get('completed_chunks',0)}/{len(PODCAST_TURNS)}|failed:{len(failed_chunks)}|eta:{current_eta}"
            if status_str != last_status_str or int(elapsed_total) % 15 < 3:
                if len(eta_snapshots) == 0 or status_str != last_status_str:
                    eta_snapshots.append({
                        "elapsed": elapsed_total,
                        "status": state,
                        "eta": current_eta,
                        "completed": status.get("completed_chunks", 0),
                        "failed": len(failed_chunks)
                    })
                    if initial_eta is None and current_eta is not None:
                        initial_eta = current_eta
                    last_status_str = status_str
            
            # Update chunk tracking
            for c in chunks:
                idx = c.get("index")
                if idx is None:
                    continue
                stat = chunk_stats[idx]
                
                if stat["first_seen"] is None:
                    stat["first_seen"] = elapsed_total
                
                cstatus = c.get("status", "unknown")
                api_elapsed = c.get("elapsed_seconds")
                if cstatus in ("done", "completed") and stat["completed_at"] is None:
                    stat["completed_at"] = elapsed_total
                    # Prefer API-reported elapsed time for accuracy
                    if api_elapsed is not None:
                        stat["elapsed"] = api_elapsed
                    else:
                        stat["elapsed"] = elapsed_total - stat["first_seen"]
                    stat["final_status"] = "done"
                elif cstatus == "failed":
                    stat["final_status"] = "failed"
                    err = c.get("error", "unknown error")
                    if stat["last_error"] != err:
                        stat["failed_count"] += 1
                        stat["last_error"] = err
            
            # Auto-retry failed chunks
            for fc in failed_chunks:
                idx = fc.get("index")
                err = fc.get("error", "unknown")
                print(f"[RETRY] Chunk {idx} failed ({err}), triggering retry at {format_duration(elapsed_total)}")
                try:
                    rresult = retry_chunk(job_id, idx)
                    retries_log.append({
                        "chunk_idx": idx,
                        "elapsed": elapsed_total,
                        "result": rresult,
                        "error": err
                    })
                    # Reset tracking for this chunk since it's being retried
                    chunk_stats[idx]["completed_at"] = None
                    chunk_stats[idx]["elapsed"] = None
                    chunk_stats[idx]["final_status"] = "retrying"
                except Exception as e:
                    print(f"[WARN] Retry failed for chunk {idx}: {e}")
                    retries_log.append({
                        "chunk_idx": idx,
                        "elapsed": elapsed_total,
                        "result": str(e),
                        "error": err
                    })
            
            # Print concise progress
            done_count = sum(1 for s in chunk_stats.values() if s["final_status"] == "done")
            fail_count = len(failed_chunks)
            print(f"[{format_duration(elapsed_total):>8}] status={state:>8} | done={done_count:>2}/{len(PODCAST_TURNS)} | failed={fail_count:>2} | eta={current_eta}")
            
            # Check termination
            if state == "done" and fail_count == 0:
                print("\n[DONE] Job completed successfully with 0 failed chunks.")
                break
            
            time.sleep(3)
        else:
            print("\n[TIMEOUT] Script reached deadline before job completion.")
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Stopped by user.")
    
    total_wall = time.time() - start_time
    final_status = get_job_status(job_id)
    
    # Build report
    print("\n")
    print("="*70)
    print("RESULTS REPORT")
    print("="*70)
    print(f"Job ID:        {job_id}")
    print(f"Total Wall Time: {format_duration(total_wall)}")
    print(f"Final Status:  {final_status.get('status', 'unknown')}")
    final_chunks = final_status.get('chunks', [])
    final_failed = [c for c in final_chunks if c.get('status') == 'failed']
    print(f"Final Failed:  {len(final_failed)}")
    print()
    
    # Per-chunk table
    print("-"*100)
    print(f"{'Idx':>4} | {'Voice':>8} | {'Status':>10} | {'Chars':>5} | {'Elapsed':>8} | {'Chars/s':>8} | {'Expected':>8} | {'Diff':>8}")
    print("-"*100)
    
    for idx in range(len(PODCAST_TURNS)):
        s = chunk_stats[idx]
        chars = s["chars"]
        elapsed = s["elapsed"]
        cps = chars / elapsed if elapsed and elapsed > 0 else 0.0
        expected = chars / 19.0
        diff = elapsed - expected if elapsed is not None else None
        
        elapsed_str = f"{elapsed:.2f}s" if elapsed is not None else "N/A"
        cps_str = f"{cps:.1f}" if elapsed is not None else "N/A"
        expected_str = f"{expected:.2f}s"
        diff_str = f"{diff:+.2f}s" if diff is not None else "N/A"
        
        print(f"{idx:>4} | {s['voice']:>8} | {s['final_status']:>10} | {chars:>5} | {elapsed_str:>8} | {cps_str:>8} | {expected_str:>8} | {diff_str:>8}")
    print("-"*100)
    print()
    
    # Retries performed
    print("-"*70)
    print("RETRIES PERFORMED")
    print("-"*70)
    if not retries_log:
        print("None.")
    else:
        for r in retries_log:
            print(f"  Chunk {r['chunk_idx']} at {format_duration(r['elapsed'])} — prior error: {r['error']} — result: {r['result']}")
    print()
    
    # ETA accuracy
    print("-"*70)
    print("ETA ACCURACY")
    print("-"*70)
    if initial_eta is not None:
        print(f"Initial ETA:       {initial_eta:.2f}s")
        print(f"Actual Total Time: {total_wall:.2f}s")
        print(f"ETA Error:         {initial_eta - total_wall:+.2f}s")
    else:
        print("No ETA was reported.")
    print()
    
    # ETA snapshots
    print("-"*70)
    print("RUNNING ETA SNAPSHOTS")
    print("-"*70)
    for snap in eta_snapshots:
        print(f"  Elapsed={format_duration(snap['elapsed']):>8}  Status={snap['status']:>8}  Done={snap['completed']:>2}/{len(PODCAST_TURNS)}  Failed={snap['failed']:>2}  ETA={snap['eta']}")
    print()
    
    # Final download URL
    print("-"*70)
    print("FINAL DOWNLOAD URL")
    print("-"*70)
    download_url = f"http://{SERVER_HOSTNAME}:8883/api/jobs/{job_id}/download?file=final.mp3"
    print(download_url)
    print()
    
    # Verify the URL is reachable from localhost perspective
    local_url = f"{BASE_URL}/api/jobs/{job_id}/download?file=final.mp3"
    try:
        head = requests.head(local_url, timeout=10)
        print(f"[VERIFY] HEAD {local_url} => {head.status_code}")
    except Exception as e:
        print(f"[VERIFY] HEAD check failed: {e}")
    
    print("="*70)
    print("END OF REPORT")
    print("="*70)

if __name__ == "__main__":
    main()
