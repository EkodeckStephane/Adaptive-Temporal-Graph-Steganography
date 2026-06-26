import re
from pathlib import Path

# Patterns
LABEL_RE = re.compile(r'\\label\{([^}]+)\}')
REF_RE = re.compile(r'\\(?:ref|cref|Cref|eqref|autoref|nameref|pageref)\{([^}]+)\}')
ENV_RE = re.compile(r'\\begin\{(table|figure|algorithm|equation|align|alignat|gather|multline|subequations)\*?\}')
END_ENV_RE = re.compile(r'\\end\{(table|figure|algorithm|equation|align|alignat|gather|multline|subequations)\*?\}')
CAPTION_RE = re.compile(r'\\caption\{')

# Directories to scan
docs = {
    'article': [
        'article/main.tex',
        *sorted(Path('article/sections').glob('*.tex')),
        *sorted(Path('article/figures').glob('*.tikz')),
    ],
    'thesis': [
        'thesis/main.tex',
        *sorted(Path('thesis/sections').glob('*.tex')),
        *sorted(Path('thesis/frontmatter').glob('*.tex')),
    ],
}

results = {}

for doc_name, files in docs.items():
    labels = {}  # label -> (file, line)
    refs = {}  # label -> list of (file, line)
    envs = []  # list of dicts
    
    for f in files:
        p = Path(f)
        if not p.exists():
            continue
        content = p.read_text(encoding='utf-8', errors='ignore')
        lines = content.splitlines()
        
        # Track environments and labels/refs line by line
        stack = []
        for i, line in enumerate(lines, 1):
            # Find environment starts
            for m in ENV_RE.finditer(line):
                env = m.group(1)
                stack.append({'env': env, 'file': str(p), 'line': i, 'has_label': False, 'has_caption': False, 'label': None})
            
            # Check for caption in current environments
            if CAPTION_RE.search(line):
                for e in stack:
                    e['has_caption'] = True
            
            # Check for label
            for m in LABEL_RE.finditer(line):
                lbl = m.group(1)
                labels[lbl] = (str(p), i)
                # Associate with innermost environment if any
                if stack:
                    stack[-1]['has_label'] = True
                    stack[-1]['label'] = lbl
            
            # Check for refs
            for m in REF_RE.finditer(line):
                ref = m.group(1)
                refs.setdefault(ref, []).append((str(p), i))
            
            # Find environment ends
            for m in END_ENV_RE.finditer(line):
                env = m.group(1)
                # Pop matching environment from stack
                for j in range(len(stack) - 1, -1, -1):
                    if stack[j]['env'] == env:
                        envs.append(stack.pop(j))
                        break
    
    # After processing, any remaining open environments (malformed)
    envs.extend(stack)
    
    # Analyze
    label_keys = set(labels.keys())
    ref_keys = set(refs.keys())
    
    unused_labels = label_keys - ref_keys
    undefined_refs = ref_keys - label_keys
    
    # Environments needing label/caption
    envs_needing_label = []
    envs_needing_caption = []
    for e in envs:
        if e['env'] in ('table', 'figure', 'algorithm'):
            if not e['has_label']:
                envs_needing_label.append(e)
            if not e['has_caption']:
                envs_needing_caption.append(e)
    
    # Equations without label are OK, but let's list those with label unused
    equation_labels_unused = []
    for e in envs:
        if e['env'] in ('equation', 'align', 'alignat', 'gather', 'multline', 'subequations'):
            if e['label'] and e['label'] in unused_labels:
                equation_labels_unused.append(e)
    
    results[doc_name] = {
        'labels': labels,
        'refs': refs,
        'envs': envs,
        'unused_labels': unused_labels,
        'undefined_refs': undefined_refs,
        'envs_needing_label': envs_needing_label,
        'envs_needing_caption': envs_needing_caption,
        'equation_labels_unused': equation_labels_unused,
    }

for doc_name, r in results.items():
    print(f"\n{'='*60}")
    print(f"DOCUMENT: {doc_name}")
    print(f"{'='*60}")
    print(f"Total labels: {len(r['labels'])}")
    print(f"Total refs: {sum(len(v) for v in r['refs'].values())}")
    print(f"Environments found: {len(r['envs'])}")
    
    env_counts = {}
    for e in r['envs']:
        env_counts[e['env']] = env_counts.get(e['env'], 0) + 1
    print(f"Environment counts: {env_counts}")
    
    if r['undefined_refs']:
        print(f"\n[ERR] Undefined references ({len(r['undefined_refs'])}):")
        for ref in sorted(r['undefined_refs']):
            for f, ln in r['refs'][ref]:
                print(f"   \\ref{{{ref}}} in {f}:{ln}")
    else:
        print("\n[OK] All references resolve to a label.")
    
    if r['unused_labels']:
        print(f"\n[WARN]  Unused labels ({len(r['unused_labels'])}):")
        for lbl in sorted(r['unused_labels']):
            f, ln = r['labels'][lbl]
            print(f"   \\label{{{lbl}}} in {f}:{ln}")
    else:
        print("\n[OK] All labels are referenced.")
    
    if r['envs_needing_label']:
        print(f"\n[ERR] Floats without label ({len(r['envs_needing_label'])}):")
        for e in r['envs_needing_label']:
            print(f"   {e['env']} in {e['file']}:{e['line']}")
    else:
        print("\n[OK] All floats have a label.")
    
    if r['envs_needing_caption']:
        print(f"\n[ERR] Floats without caption ({len(r['envs_needing_caption'])}):")
        for e in r['envs_needing_caption']:
            print(f"   {e['env']} in {e['file']}:{e['line']}")
    else:
        print("\n[OK] All floats have a caption.")
    
    if r['equation_labels_unused']:
        print(f"\n[WARN]  Equations labeled but never referenced ({len(r['equation_labels_unused'])}):")
        for e in r['equation_labels_unused']:
            print(f"   \\label{{{e['label']}}} in {e['file']}:{e['line']}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for doc_name, r in results.items():
    issues = (
        len(r['undefined_refs']) + len(r['envs_needing_label']) +
        len(r['envs_needing_caption'])
    )
    if issues == 0 and len(r['unused_labels']) == 0:
        print(f"{doc_name}: [OK] clean")
    else:
        print(f"{doc_name}: {issues} errors, {len(r['unused_labels'])} unused labels")
