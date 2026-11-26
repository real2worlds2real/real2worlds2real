"""Deliver retained user/runtime updates into an active Codex turn."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

from .artifacts import atomic_json


def queue_instruction(store, message, *, source="user"):
    if source not in {"user", "runtime"} or not message.strip():
        raise ValueError("A user/runtime instruction needs nonempty text")
    identifier = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]
    record = {"id": identifier, "source": source, "message": message}
    atomic_json(store.root / "instructions" / f"{identifier}.json", record)
    return record


def deliver_pending(store, runtime, transport, thread_id, turn_id):
    receipt_path = runtime / "instruction_receipts.json"
    receipts = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
    deferred_path = runtime / "instruction_deferred.json"
    deferred = json.loads(deferred_path.read_text()) if deferred_path.is_file() else {}
    delivered = []
    for path in sorted((store.root / "instructions").glob("*.json")):
        instruction = json.loads(path.read_text())
        if instruction["id"] in receipts:
            continue
        if deferred.get(instruction["id"]) == turn_id:
            continue
        text = f"New {instruction['source']} update for the current task:\n{instruction['message']}"
        try:
            response = transport.request(
                "turn/steer",
                {
                    "threadId": thread_id,
                    "expectedTurnId": turn_id,
                    "input": [{"type": "text", "text": text}],
                },
            )
        except RuntimeError as error:
            # The turn may have completed while its final event is still queued.
            # Retain the update and retry on the next turn; never mark it delivered.
            if "Codex RPC turn/steer rejected (code=-32600)" not in str(error):
                raise
            deferred[instruction["id"]] = turn_id
            atomic_json(deferred_path, deferred)
            store.append_event({"type": "instruction_deferred", "instruction_id": instruction["id"],
                                "turn_id": turn_id, "reason": "turn/steer rejected; retain for next turn"})
            continue
        if response.get("turnId") != turn_id:
            raise RuntimeError("Instruction was steered to a different turn")
        receipts[instruction["id"]] = {
            "thread_id": thread_id,
            "turn_id": turn_id,
            "source": instruction["source"],
        }
        atomic_json(receipt_path, receipts)
        store.append_event(
            {
                "type": "instruction_delivered",
                "instruction_id": instruction["id"],
                "source": instruction["source"],
                "turn_id": turn_id,
            }
        )
        delivered.append(instruction)
    return delivered


def pending_for_acceptance_reply(store, runtime, thread_id, turn_id):
    """Deliver a deferred update in the rejection reply before accepting anything."""
    path = runtime / "instruction_receipts.json"
    receipts = json.loads(path.read_text()) if path.is_file() else {}
    result = []
    for instruction_path in sorted((store.root / "instructions").glob("*.json")):
        row = json.loads(instruction_path.read_text())
        if row["id"] in receipts:
            continue
        result.append(row)
        receipts[row["id"]] = {"thread_id": thread_id, "turn_id": turn_id,
                                "source": row["source"], "channel": "acceptance_rejection_reply"}
        store.append_event({"type": "instruction_delivered", "instruction_id": row["id"],
                            "source": row["source"], "turn_id": turn_id,
                            "channel": "acceptance_rejection_reply"})
    if result:
        atomic_json(path, receipts)
    return result
