from dataclasses import dataclass, field


@dataclass
class Block:
    """A named slot inside working context. Analogous to Letta's Block schema."""

    label: str
    value: str = ""
    limit: int = 2000
    description: str = ""

    def set(self, new_value: str) -> None:
        if len(new_value) > self.limit:
            raise ValueError(f"Block '{self.label}' exceeds char limit {self.limit} (got {len(new_value)})")
        self.value = new_value


@dataclass
class CoreMemory:
    """Working context: a set of named blocks rendered into the system prompt."""

    blocks: dict[str, Block] = field(default_factory=dict)

    def add(self, block: Block) -> None:
        self.blocks[block.label] = block

    def get(self, label: str) -> Block:
        if label not in self.blocks:
            raise KeyError(f"No memory block with label '{label}'")
        return self.blocks[label]

    def append(self, label: str, content: str) -> str:
        b = self.get(label)
        b.set((b.value + "\n" + content).strip() if b.value else content)
        return b.value

    def replace(self, label: str, old: str, new: str) -> str:
        b = self.get(label)
        if not old:
            raise ValueError(
                f"old_content must be non-empty; use core_memory_append to add to an empty block "
                f"(block '{label}' currently has {len(b.value)} chars)"
            )
        if old not in b.value:
            raise ValueError(f"'{old}' not found in block '{label}'")
        b.set(b.value.replace(old, new))
        return b.value

    def compile(self) -> str:
        """Render all blocks into a text fragment for the system prompt."""
        if not self.blocks:
            return "<core_memory />"
        parts = ["<core_memory>"]
        for label, b in self.blocks.items():
            desc = f' description="{b.description}"' if b.description else ""
            chars = f'chars="{len(b.value)}/{b.limit}"'
            if not b.value:
                # Self-close empty blocks so the LLM doesn't mistake a placeholder
                # string like "(empty)" for the block's actual content.
                parts.append(f'  <block label="{label}" {chars} empty="true"{desc}/>')
                continue
            parts.append(f'  <block label="{label}" {chars}{desc}>')
            parts.append(b.value)
            parts.append(f"  </block>")
        parts.append("</core_memory>")
        return "\n".join(parts)
