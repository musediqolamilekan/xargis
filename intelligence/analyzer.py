import os
import re
import warnings
from collections import Counter
from typing import Any, Optional

# Suppress tree-sitter 0.21 / tree-sitter-languages compatibility FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="tree_sitter")


from .models import (
    FileAnalysis,
    SymbolLocation,
    EXTENSION_TO_LANGUAGE,
    SYMBOL_QUERIES,
    TAGS_QUERIES,
)
from .query_engine import tokenize_identifiers

#Skip directories that are commonly not relevant for source code analysis
SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", ".env",
    "dist", "build", ".next", ".nuxt", "target", "vendor",
    ".idea", ".vscode", ".mypy_cache", ".pytest_cache",
    ".repo_cache", ".asre",
}

#Skip files with these extensions (case-insensitive)
SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",
    ".mp4", ".mp3", ".wav", ".avi", ".mov",
    ".zip", ".tar", ".gz", ".bz2", ".rar", ".7z",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".ttf", ".woff", ".woff2", ".eot",
    ".lock", ".min.js", ".min.css",
    ".pyc", ".pyo", ".class",
    ".bin", ".exe", ".so", ".dll", ".dylib",
    ".db", ".sqlite", ".sqlite3",
}

TODO_PATTERN = re.compile(r"(TODO|FIXME|HACK|XXX|BUG|NOTE):\s*(.*)", re.IGNORECASE)

_CALL_KEYWORDS = frozenset({
    "if", "for", "while", "switch", "catch", "return", "sizeof", "typeof",
    "instanceof", "print", "len", "str", "int", "float", "bool", "list",
    "dict", "set", "tuple", "range", "enumerate", "zip", "min", "max",
    "sum", "sorted", "reversed", "any", "all", "isinstance", "getattr",
    "setattr", "hasattr", "super",
})
_CALL_FALLBACK_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)\s*\(")

_API_ROUTE_PATTERNS = (
    # Python FastAPI / Flask / Litestar / Sanic decorators
    re.compile(
        r"@(?:app|router|bp|blueprint|api)\.(get|post|put|delete|patch|options|head|route|websocket)\s*\(\s*['\"]([^'\"]+)['\"]",
        re.IGNORECASE,
    ),
    # Express / Fastify / Hono / Koa JS/TS routes
    re.compile(
        r"\b(?:app|router|server)\.(get|post|put|delete|patch)\s*\(\s*['\"](/[^'\"]*)['\"]",
        re.IGNORECASE,
    ),
    # Laravel / PHP routes
    re.compile(
        r"\bRoute::(get|post|put|delete|patch|any)\s*\(\s*['\"]([^'\"]+)['\"]",
        re.IGNORECASE,
    ),
    # Java Spring Boot annotations
    re.compile(
        r"@(GetMapping|PostMapping|PutMapping|DeleteMapping|PatchMapping|RequestMapping)\s*\(\s*(?:value\s*=\s*)?['\"]([^'\"]+)['\"]",
    ),
    # Go net/http, Gin, Echo, Fiber
    re.compile(
        r"\b(?:http\.HandleFunc|r|router|e|app)\.(GET|POST|PUT|DELETE|PATCH|HandleFunc)\s*\(\s*['\"](/[^'\"]*)['\"]",
    ),
)

_DB_SQL_RE = re.compile(
    r"\b(SELECT\s+.+?\s+FROM\s+[A-Za-z0-9_`\"'.]+|INSERT\s+INTO\s+[A-Za-z0-9_`\"'.]+|UPDATE\s+[A-Za-z0-9_`\"'.]+\s+SET|DELETE\s+FROM\s+[A-Za-z0-9_`\"'.]+|CREATE\s+TABLE(?:\s+IF\s+NOT\s+EXISTS)?\s+[A-Za-z0-9_`\"'.]+)",
    re.IGNORECASE,
)
_DB_ORM_PATTERNS = (
    re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*\.objects\.(?:filter|get|create|update|delete|all|exclude|raw))\s*\("),
    re.compile(r"\b(session\.(?:query|execute|add|commit|scalars)|db\.(?:session|execute|query|collection))\b"),
    re.compile(r"\b(prisma\.[A-Za-z_][A-Za-z0-9_]*\.(?:findUnique|findMany|findFirst|create|update|delete|upsert))\s*\("),
    re.compile(r"\b(DB::(?:table|select|insert|update|delete|statement)|[A-Z][A-Za-z0-9_]*::(?:where|find|create|query))\s*\("),
    re.compile(r"\b(cursor\.execute|conn\.execute|sqlite3\.connect|psycopg2\.connect|asyncpg\.connect|redis\.[A-Za-z_]+)\s*\("),
)

_SERVICE_CALL_PATTERNS = (
    re.compile(r"\b(requests\.(?:get|post|put|delete|patch|request)|httpx\.(?:get|post|put|delete|patch|AsyncClient|Client))\s*\("),
    re.compile(r"\b(urllib\.request\.(?:urlopen|Request)|aiohttp\.ClientSession)\b"),
    re.compile(r"\b(axios\.(?:get|post|put|delete|patch|request)|fetch)\s*\("),
    re.compile(r"\b(grpc\.(?:insecure_channel|secure_channel)|boto3\.(?:client|resource))\s*\("),
    re.compile(r"\b(celery\.Celery|send_task|\.delay|\.apply_async|kafka\.KafkaProducer|kafka\.KafkaConsumer)\b"),
)

_CONFIG_KEY_PATTERNS = (
    re.compile(r"\bos\.environ(?:\.get)?\s*[\[\(]\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]"),
    re.compile(r"\bos\.getenv\s*\(\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]"),
    re.compile(r"\bprocess\.env\.([A-Za-z_][A-Za-z0-9_]*)\b"),
    re.compile(r"\b(?:env|config)\s*\(\s*['\"]([A-Za-z_][A-Za-z0-9_.]*)['\"]"),
    re.compile(r"\bsettings\.([A-Z][A-Z0-9_]{2,})\b"),
)


def extract_deep_code_signals(
    parser,
    language_name: Optional[str],
    source_bytes: bytes,
    source_str: str,
) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    """
    Extract Section B deep code intelligence signals:
      (calls, api_endpoints, db_interactions, service_calls, config_keys)
    """
    calls: list[str] = []
    seen_calls: set[str] = set()

    if parser is not None:
        try:
            tree = parser.parse(source_bytes)
            call_node_types = {
                "call",
                "call_expression",
                "method_invocation",
                "function_call_expression",
                "member_call_expression",
            }
            stack = [tree.root_node]
            while stack:
                node = stack.pop()
                if node.type in call_node_types:
                    fn_node = (
                        node.child_by_field_name("function")
                        or node.child_by_field_name("name")
                        or (node.children[0] if node.children else None)
                    )
                    if fn_node is not None:
                        raw_fn = extract_node_text(fn_node, source_bytes).strip()
                        if raw_fn and len(raw_fn) <= 80 and "\n" not in raw_fn:
                            short_name = raw_fn.split("(")[0].strip()
                            base_tok = short_name.rsplit(".", 1)[-1]
                            if base_tok not in _CALL_KEYWORDS and short_name not in seen_calls:
                                seen_calls.add(short_name)
                                calls.append(short_name)
                for child in reversed(node.children):
                    stack.append(child)
        except Exception:
            pass

    if not calls:
        for match in _CALL_FALLBACK_RE.finditer(source_str):
            candidate = match.group(1)
            base_tok = candidate.rsplit(".", 1)[-1]
            if base_tok not in _CALL_KEYWORDS and candidate not in seen_calls:
                seen_calls.add(candidate)
                calls.append(candidate)

    api_endpoints: list[str] = []
    for pattern in _API_ROUTE_PATTERNS:
        for match in pattern.finditer(source_str):
            method = match.group(1).upper().replace("MAPPING", "")
            route = match.group(2)
            entry = f"{method} {route}"
            if entry not in api_endpoints:
                api_endpoints.append(entry)

    db_interactions: list[str] = []
    for match in _DB_SQL_RE.finditer(source_str):
        stmt = " ".join(match.group(1).split())[:80]
        if stmt not in db_interactions:
            db_interactions.append(stmt)
    for pattern in _DB_ORM_PATTERNS:
        for match in pattern.finditer(source_str):
            op = match.group(1)
            if op not in db_interactions:
                db_interactions.append(op)

    service_calls: list[str] = []
    for pattern in _SERVICE_CALL_PATTERNS:
        for match in pattern.finditer(source_str):
            svc = match.group(1)
            if svc not in service_calls:
                service_calls.append(svc)

    config_keys: list[str] = []
    for pattern in _CONFIG_KEY_PATTERNS:
        for match in pattern.finditer(source_str):
            key = match.group(1)
            if key not in config_keys:
                config_keys.append(key)

    return calls[:80], api_endpoints[:40], db_interactions[:40], service_calls[:40], config_keys[:40]


def get_parser(language_name: str):
    """Lazily loads a Tree-sitter parser for the given language."""
    try:
        from tree_sitter_languages import get_parser as _get_parser  # pyrefly: ignore[missing-import]
        return _get_parser(language_name)
    except Exception:
        return None


def get_language(language_name: str):
    """Lazily loads a Tree-sitter Language object."""
    try:
        from tree_sitter_languages import get_language as _get_language  # pyrefly: ignore[missing-import]
        return _get_language(language_name)
    except Exception:
        return None


def extract_node_text(node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def run_query(parser, language_name: str, source_bytes: bytes, query_str: str) -> list[str]:
    """Runs a Tree-sitter query on the source and returns matched node texts."""
    return [
        location.name
        for location in run_symbol_query(parser, language_name, source_bytes, query_str)
    ]


def run_symbol_query(parser, language_name: str, source_bytes: bytes, query_str: str) -> list[SymbolLocation]:
    """Runs a legacy Tree-sitter query and retains each matched source range."""
    try:
        lang = get_language(language_name)
        if lang is None:
            return []
        tree = parser.parse(source_bytes)
        captures = lang.query(query_str).captures(tree.root_node)
        locations: list[SymbolLocation] = []
        seen: set[tuple[str, str, int, int]] = set()

        for item in captures:
            if isinstance(item, tuple):
                node, capture_name = item
                kind = capture_name.lstrip("@").replace("_", "-")
            else:
                node = item
                kind = "symbol"

            target_node = node
            if kind in {"func", "cls", "interface", "struct", "enum", "trait", "module"}:
                target_node = node.parent or node
            location = SymbolLocation(
                name=extract_node_text(node, source_bytes),
                kind=kind,
                start_line=target_node.start_point[0] + 1,
                end_line=max(target_node.start_point[0] + 1, target_node.end_point[0] + 1),
            )
            identity = (location.name, location.kind, location.start_line, location.end_line)
            if identity not in seen:
                seen.add(identity)
                locations.append(location)
        return locations
    except Exception:
        return []


def run_tags_query(parser, language_name: str, source_bytes: bytes) -> list[SymbolLocation]:
    """Extract definitions using Tree-sitter's standardized @definition tags.

    Returns an empty list for an unknown language or invalid tag query. Callers
    may then use their explicit legacy query as a compatibility fallback.
    """
    query_str = TAGS_QUERIES.get(language_name)
    if not query_str:
        return []

    try:
        lang = get_language(language_name)
        if lang is None:
            return []
        captures = lang.query(query_str).captures(parser.parse(source_bytes).root_node)
        names: list[tuple[int, int, str]] = []
        definitions: list[tuple[str, Any]] = []

        for item in captures:
            if not isinstance(item, tuple):
                continue
            node, capture_name = item
            if capture_name == "name":
                names.append((node.start_byte, node.end_byte, extract_node_text(node, source_bytes)))
            elif capture_name.startswith("definition."):
                kind = capture_name.split(".", 1)[1]
                definitions.append((kind, node))

        locations: list[SymbolLocation] = []
        for kind, definition_node in definitions:
            name_matches = [
                name
                for start_byte, end_byte, name in names
                if definition_node.start_byte <= start_byte and end_byte <= definition_node.end_byte
            ]
            name = name_matches[0] if name_matches else ""
            if not name:
                continue
            locations.append(SymbolLocation(
                name=name,
                kind=kind,
                start_line=definition_node.start_point[0] + 1,
                end_line=max(definition_node.start_point[0] + 1, definition_node.end_point[0] + 1),
            ))
        return locations
    except Exception:
        return []


def run_structural_ast_fallback(parser, language_name: str, source_bytes: bytes) -> tuple[list[SymbolLocation], list[dict]]:
    """Capture language-neutral structural nodes and parse errors."""
    try:
        tree = parser.parse(source_bytes)
        locations: list[SymbolLocation] = []
        parse_errors: list[dict] = []
        seen: set[tuple[str, int, int]] = set()

        def visit(node) -> None:
            node_type = node.type.lower()
            kind: Optional[str] = None
            if node_type in {"error"} or "error" in node_type:
                kind = "ERROR"
            elif "declaration" in node_type or "definition" in node_type:
                kind = "declaration"
            elif "assignment" in node_type or "initializer" in node_type:
                kind = "assignment"
            elif node_type.endswith("block") or node_type in {"block", "statement_block", "compound_statement"}:
                kind = "block"

            if kind:
                identity = (kind, node.start_point[0] + 1, node.end_point[0] + 1)
                if kind == "ERROR":
                    parse_errors.append({
                        "kind": kind,
                        "start_line": node.start_point[0] + 1,
                        "end_line": max(node.start_point[0] + 1, node.end_point[0] + 1),
                        "text": extract_node_text(node, source_bytes)[:500],
                    })
                elif identity not in seen:
                    seen.add(identity)
                    name = ""
                    try:
                        name_node = node.child_by_field_name("name")
                        if name_node:
                            name = extract_node_text(name_node, source_bytes)
                    except Exception:
                        pass
                    if not name:
                        match = re.search(r"[A-Za-z_]\w*", extract_node_text(node, source_bytes))
                        name = match.group(0) if match else node_type
                    locations.append(SymbolLocation(
                        name=name,
                        kind=kind,
                        start_line=node.start_point[0] + 1,
                        end_line=max(node.start_point[0] + 1, node.end_point[0] + 1),
                    ))

            for child in node.children:
                visit(child)

        visit(tree.root_node)
        return locations, parse_errors
    except Exception:
        return [], []


def get_symbol_locations(parser, language_name: str, source_bytes: bytes) -> list[SymbolLocation]:
    """Return tagged definitions, falling back to explicit legacy queries."""
    tagged = run_tags_query(parser, language_name, source_bytes)
    if tagged:
        return tagged

    locations: list[SymbolLocation] = []
    for query_name, query_str in SYMBOL_QUERIES.get(language_name, {}).items():
        if query_name in {"imports", "from_imports", "includes"}:
            continue
        locations.extend(run_symbol_query(parser, language_name, source_bytes, query_str))
    return locations


def extract_todos(source: str, file_path: str) -> list[dict]:
    """Finds TODO/FIXME style comments in source code."""
    todos = []
    for i, line in enumerate(source.splitlines(), start=1):
        match = TODO_PATTERN.search(line)
        if match:
            todos.append({
                "type": match.group(1).upper(),
                "message": match.group(2).strip(),
                "file": file_path,
                "line": i,
            })
    return todos


def should_skip(path: str) -> bool:
    """Returns True if a file or directory should be skipped."""
    name = os.path.basename(path)
    if name in SKIP_DIRS:
        return True
    ext = os.path.splitext(name)[1].lower()
    if ext in SKIP_EXTENSIONS:
        return True
    # Skip minified files
    if ".min." in name:
        return True
    return False


def analyze_file(file_path: str, repo_root: str) -> Optional[FileAnalysis]:
    """Analyzes a single file using Tree-sitter and heuristics."""
    rel_path = os.path.relpath(file_path, repo_root)

    if should_skip(file_path):
        return None

    ext = os.path.splitext(file_path)[1].lower()
    language_name = EXTENSION_TO_LANGUAGE.get(ext)

    try:
        size_bytes = os.path.getsize(file_path)
        # Skip very large files (>1MB)
        if size_bytes > 1_000_000:
            return None

        with open(file_path, "rb") as f:
            source_bytes = f.read()

        source_str = source_bytes.decode("utf-8", errors="replace")
        lines = len(source_str.splitlines())
        todos = extract_todos(source_str, rel_path)

        analysis = FileAnalysis(
            path=rel_path,
            language=language_name or "unknown",
            lines=lines,
            size_bytes=size_bytes,
            todos=todos,
            lexical_terms=dict(Counter(tokenize_identifiers(source_str))),
            analysis_tier="lexical",
        )

        parser = get_parser(language_name) if language_name else None
        calls, api_endpoints, db_interactions, service_calls, config_keys = extract_deep_code_signals(
            parser,
            language_name,
            source_bytes,
            source_str,
        )
        analysis.calls = calls
        analysis.api_endpoints = api_endpoints
        analysis.db_interactions = db_interactions
        analysis.service_calls = service_calls
        analysis.config_keys = config_keys

        if language_name:
            if parser:
                queries = SYMBOL_QUERIES.get(language_name, {})

                if "functions" in queries:
                    analysis.functions = run_query(parser, language_name, source_bytes, queries["functions"])
                if "classes" in queries:
                    analysis.classes = run_query(parser, language_name, source_bytes, queries["classes"])
                if "imports" in queries:
                    analysis.imports += run_query(parser, language_name, source_bytes, queries["imports"])
                if "from_imports" in queries:
                    analysis.imports += run_query(parser, language_name, source_bytes, queries["from_imports"])
                if "interfaces" in queries:
                    analysis.interfaces = run_query(parser, language_name, source_bytes, queries["interfaces"])
                if "structs" in queries:
                    analysis.structs = run_query(parser, language_name, source_bytes, queries["structs"])

                locations = get_symbol_locations(parser, language_name, source_bytes)
                if locations:
                    analysis.analysis_tier = "tags"
                    analysis.symbol_locations = locations
                else:
                    structural, parse_errors = run_structural_ast_fallback(parser, language_name, source_bytes)
                    analysis.analysis_tier = "structural"
                    analysis.symbol_locations = structural
                    analysis.parse_errors = parse_errors

        return analysis

    except OSError as e:
        return FileAnalysis(
            path=rel_path,
            language=language_name or "unknown",
            lines=0,
            size_bytes=0,
            error=str(e),
        )


def walk_repository(repo_root: str) -> list[str]:
    """Walks the repository, returning all non-skipped file paths."""
    all_files = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        # Prune skipped directories in-place to avoid descending into them
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            if not should_skip(full_path):
                all_files.append(full_path)

    return all_files
