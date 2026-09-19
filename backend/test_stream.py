"""
Phase 1 — Real-time test client for the /stream WebSocket endpoint.

Run from the backend folder:
    cd c:\\projects\\hacksporta\\backend
    .\\venv\\Scripts\\python test_stream.py

Face results arrive every ~0.5 sec.
Audio results arrive whenever you finish speaking (VAD detects silence).
Press Ctrl+C to stop.
"""

import sys
import json
import asyncio
import websockets


WS_URL = "ws://localhost:8000/stream?camera=0"

GREEN  = "\033[92m"
BLUE   = "\033[94m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


async def stream():
    print(f"\n{BOLD}🔗 Connecting to {WS_URL} …{RESET}\n")
    try:
        async with websockets.connect(WS_URL, ping_interval=None) as ws:
            print(f"{BOLD}✅ Connected — real-time analysis running.{RESET}")
            print("   • Face emotion updates every ~0.5 sec")
            print("   • Speech/transcript updates after each utterance (VAD-based)")
            print("   Press Ctrl+C to stop.\n")
            print("─" * 60)

            try:
                while True:
                    raw = await ws.recv()
                    data = json.loads(raw)
                    ptype = data.get("type", "unknown")

                    if ptype == "face":
                        face = data.get("data", {})
                        if "error" in face:
                            print(f"{RED}🎭 Face   : ERROR — {face['error']}{RESET}")
                        else:
                            emo = face.get("emotion", "?")
                            conf = face.get("confidence", 0)
                            lat = face.get("latency_ms", "?")
                            bar = "█" * int(conf * 20)
                            print(f"{GREEN}🎭 Face   : {BOLD}{emo:<12}{RESET}{GREEN} {conf:.2f}  {bar}  ({lat}ms){RESET}")

                    elif ptype == "audio":
                        if "error" in data:
                            print(f"{RED}🎤 Audio  : ERROR — {data['error']}{RESET}")
                        else:
                            speech = data.get("speech", {})
                            tx     = data.get("transcript", {})
                            emo    = speech.get("emotion", "?")
                            conf   = speech.get("confidence", 0)
                            lat    = speech.get("latency_ms", "?")
                            text   = tx.get("transcript", "—")
                            bar    = "█" * int(conf * 20)
                            print(f"{BLUE}🎤 Speech : {BOLD}{emo:<12}{RESET}{BLUE} {conf:.2f}  {bar}  ({lat}ms){RESET}")
                            print(f'{YELLOW}📝 Text   : "{text}"{RESET}')
                            print("─" * 60)
            except (asyncio.CancelledError, KeyboardInterrupt):
                try:
                    await ws.send("stop")
                except Exception:
                    pass

    except (ConnectionRefusedError, OSError):
        print(f"{RED}❌ Cannot connect — is uvicorn running on port 8000?{RESET}")
        print("   Run: cd backend && .\\venv\\Scripts\\python -m uvicorn main:app --port 8000")
    except websockets.exceptions.ConnectionClosedOK:
        print(f"\n{BOLD}🛑 Stream ended cleanly.{RESET}")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"{RED}❌ Connection closed: {e}{RESET}")


if __name__ == "__main__":
    if "--gui" in sys.argv or "-g" in sys.argv:
        from live_face_view import LiveFaceViewer
        viewer = LiveFaceViewer(camera_index=0)
        viewer.run()
    else:
        try:
            asyncio.run(stream())
        except KeyboardInterrupt:
            print(f"\n{BOLD}🛑 Stopped.{RESET}")
