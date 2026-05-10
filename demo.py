"""Interactive demo: chat with a MemGPT-mini agent."""
import asyncio

from memgpt.agent import MemGPTAgent
from memgpt.config import load_config
from memgpt.db import Database


BANNER = """
MemGPT-mini demo. Commands:
  /quit           exit
  /blocks         show core memory blocks
  /messages       show in-context message count + roles
  /archive <txt>  manually insert into archival
  /search <q>     manually search archival
Type anything else to talk to the agent.
"""


async def main() -> None:
    cfg = load_config()
    db = Database(cfg)
    await db.init_schema()

    agent = MemGPTAgent(cfg=cfg, db=db)
    print(BANNER)
    print(f"[agent_id={agent.agent_id} model={cfg.model} ctx={cfg.context_window}]")

    while True:
        try:
            line = input("\nyou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line == "/quit":
            break
        if line == "/blocks":
            for label, b in agent.core_memory.blocks.items():
                print(f"  [{label}] ({len(b.value)}/{b.limit}) {b.value!r}")
            continue
        if line == "/messages":
            roles = [m.get("role") for m in agent.in_context]
            print(f"  in_context={len(roles)} roles={roles}")
            continue
        if line.startswith("/archive "):
            pid = await agent.archival.insert(line[len("/archive "):])
            print(f"  inserted passage {pid}")
            continue
        if line.startswith("/search "):
            results = await agent.archival.search(line[len("/search "):])
            for r in results:
                print(f"  ({r.score:.2f}) {r.text}")
            continue

        reply = await agent.step(line)
        print(f"\nagent > {reply}")


if __name__ == "__main__":
    asyncio.run(main())
