#!/usr/bin/env python3
import re
import shutil
from pathlib import Path

# List of files and their patch actions
PATCHES = [
    {
        "path": "./nesymres/src/nesymres/dclasses.py",
        "actions": [
            {
                "name": "Insert `field` import",
                "type": "insert_after",
                "match": r"^from dataclasses import dataclass",
                "text": "from dataclasses import field",
            },
            {
                "name": "Replace mutable default with default_factory",
                "type": "replace_regex",
                "pattern": r"bfgs:\s*BFGSParams\s*=\s*BFGSParams\(\)",
                "repl": "bfgs: BFGSParams = field(default_factory=BFGSParams)",
            },
        ],
    },
    {
        "path": "./symbolicregression/optim.py",
        "actions": [
            {
                "name": "Replace getargspec → getfullargspec",
                "type": "replace_simple",
                "old": "inspect.getargspec",
                "new": "inspect.getfullargspec",
            },
        ],
    },
    {
        "path": "./symbolicregression/trainer.py",
        "actions": [
            {
                "name": "Replace np.infty → np.inf",
                "type": "replace_simple",
                "old": "np.infty",
                "new": "np.inf",
            },
        ],
    },
    {
        "path": "./symbolicregression/e2e_model.py",
        "actions": [
            {
                "name": "Add weights_only=False to torch.load",
                "type": "replace_simple",
                "old": "self.model = torch.load('./symbolicregression/weights/model.pt')",
                "new": "self.model = torch.load('./symbolicregression/weights/model.pt', weights_only=False)",
            },
        ],
    },
    {
        "path": "./symbolicregression/model/sklearn_wrapper.py",
        "actions": [
            {
                "name": "Replace np.infty → np.inf",
                "type": "replace_simple",
                "old": "np.infty",
                "new": "np.inf",
            },
        ],
    },
]


def backup_file(p: Path):
    bak = p.with_suffix(p.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(p, bak)
    return bak


def apply_patches():
    for entry in PATCHES:
        p = Path(entry["path"])
        if not p.exists():
            print(f"⚠️  File not found: {p}")
            continue

        bak = backup_file(p)
        text = p.read_text()
        original = text

        print(f"\nPatching {p} (backup at {bak.name}):")
        for act in entry["actions"]:
            if act["type"] == "insert_after":
                # only insert if not already present:
                if act["text"] in text:
                    print(f"  • [skip] {act['name']} (already present)")
                else:
                    pattern = re.compile(act["match"], re.MULTILINE)
                    text, n = pattern.subn(rf"\g<0>\n{act['text']}", text, count=1)
                    print(f"  • [insert] {act['name']}: {n} insertion")
            elif act["type"] == "replace_regex":
                text, n = re.subn(act["pattern"], act["repl"], text)
                print(f"  • [replace] {act['name']}: {n} replacements")
            elif act["type"] == "replace_simple":
                n = text.count(act["old"])
                text = text.replace(act["old"], act["new"])
                print(f"  • [replace] {act['name']}: {n} replacements")
            else:
                print(f"  • [!] Unknown action type: {act['type']}")

        # Write only if changed
        if text != original:
            p.write_text(text)
            print(f"✅  Written patched file: {p.name}")
        else:
            print("⚠️  No changes detected, file left unmodified.")


if __name__ == "__main__":
    apply_patches()
    print("\n🎉 All patches applied. You can now run your code.")
