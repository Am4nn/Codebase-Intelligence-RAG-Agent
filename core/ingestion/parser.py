import ast
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import ast
import re


class CodeParser:
    """Parse code and extract meaningful chunks with context

    This file implements simple, heuristic parsers for several languages (Python,
    JS/TS, Java) and falls back to returning the whole file as a single chunk.

    The heuristics are intentionally lightweight to avoid heavy dependencies while
    providing useful chunks (functions/classes/methods) for common project types.
    """

    @staticmethod
    def _extract_project_info(file_path: Path, repo_root: Optional[Path] = None) -> Dict[str, Optional[str]]:
        """Try to detect a project under a 'projects' folder and provide helpful metadata."""
        project_name = None
        project_relpath = None

        parts = [p for p in file_path.parts]
        # Look for a 'projects' folder in the path and take the next segment as project name
        try:
            idx = next(i for i, p in enumerate(parts) if p.lower() == 'projects')
            if idx + 1 < len(parts):
                project_name = parts[idx + 1]
                # build relative path from project root
                project_root = Path(*parts[: idx + 2])
                project_relpath = str(file_path.relative_to(project_root))
        except StopIteration:
            # not inside a projects/ folder
            pass

        # If repo_root provided and project wasn't detected, try to locate a project as the folder under repo/data/projects
        if project_name is None and repo_root:
            data_projects = repo_root / 'data' / 'projects'
            try:
                p = file_path.relative_to(data_projects)
                # first part of p is project name
                project_name = p.parts[0]
                project_relpath = str(p.relative_to(project_name)) if len(p.parts) > 1 else ''
            except Exception:
                pass

        return {
            'project_name': project_name,
            'project_relative_path': project_relpath,
        }

    @staticmethod
    def _find_block_bounds(lines: List[str], start_line: int, open_char: str = '{', close_char: str = '}') -> int:
        """Find the closing line index for a brace-delimited block starting at start_line.

        Returns the index (exclusive) of the line after the closing brace; if not
        found, returns the length of lines.
        """
        depth = 0
        started = False
        for i in range(start_line, len(lines)):
            line = lines[i]
            for ch in line:
                if ch == open_char:
                    depth += 1
                    started = True
                elif ch == close_char:
                    depth -= 1
            if started and depth <= 0:
                return i + 1
        return len(lines)

    @staticmethod
    def parse_python_file(file_path: str, content: str, repo_root: Optional[str] = None) -> List[Tuple[str, Dict[str, Any]]]:
        chunks: List[Tuple[str, Dict[str, Any]]] = []
        p = Path(file_path)
        repo = Path(repo_root) if repo_root else None
        project_info = CodeParser._extract_project_info(p, repo)

        try:
            tree = ast.parse(content)
            lines = content.split('\n')

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    start_line = node.lineno - 1
                    end_line = node.end_lineno or len(lines)
                    func_code = '\n'.join(lines[start_line:end_line])
                    metadata = {
                        'file_path': str(p),
                        'type': 'function',
                        'name': node.name,
                        'start_line': start_line,
                        'end_line': end_line,
                        **project_info,
                    }
                    chunks.append((func_code, metadata))
                elif isinstance(node, ast.ClassDef):
                    start_line = node.lineno - 1
                    end_line = node.end_lineno or len(lines)
                    class_code = '\n'.join(lines[start_line:end_line])
                    methods = [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
                    metadata = {
                        'file_path': str(p),
                        'type': 'class',
                        'name': node.name,
                        'methods': methods,
                        'start_line': start_line,
                        'end_line': end_line,
                        **project_info,
                    }
                    chunks.append((class_code, metadata))
        except SyntaxError:
            metadata = {
                'file_path': str(p),
                'type': 'file',
                'name': p.stem,
                **project_info,
            }
            chunks.append((content, metadata))
        return chunks

    @staticmethod
    def parse_js_ts_file(file_path: str, content: str, repo_root: Optional[str] = None) -> List[Tuple[str, Dict[str, Any]]]:
        """Heuristic JS/TS parser that extracts functions, arrow functions, and classes."""
        chunks: List[Tuple[str, Dict[str, Any]]] = []
        p = Path(file_path)
        repo = Path(repo_root) if repo_root else None
        project_info = CodeParser._extract_project_info(p, repo)

        lines = content.split('\n')
        fn_re = re.compile(r'^\s*(?:export\s+)?function\s+([A-Za-z0-9_]+)\s*\(')
        arrow_re = re.compile(r'^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z0-9_]+)\s*=\s*(?:async\s*)?\(?.*?\)?\s*=>')
        class_re = re.compile(r'^\s*(?:export\s+)?class\s+([A-Za-z0-9_]+)')

        for i, line in enumerate(lines):
            m = fn_re.match(line)
            if m:
                name = m.group(1)
                end = CodeParser._find_block_bounds(lines, i)
                code = '\n'.join(lines[i:end])
                metadata = {
                    'file_path': str(p),
                    'type': 'function',
                    'name': name,
                    'start_line': i,
                    'end_line': end,
                    'language': 'js',
                    **project_info,
                }
                chunks.append((code, metadata))
                continue

            m = arrow_re.match(line)
            if m:
                name = m.group(1)
                # try to capture expression body or block
                if '{' in line:
                    end = CodeParser._find_block_bounds(lines, i)
                else:
                    end = i + 1
                code = '\n'.join(lines[i:end])
                metadata = {
                    'file_path': str(p),
                    'type': 'function',
                    'name': name,
                    'start_line': i,
                    'end_line': end,
                    'language': 'js',
                    **project_info,
                }
                chunks.append((code, metadata))
                continue

            m = class_re.match(line)
            if m:
                name = m.group(1)
                end = CodeParser._find_block_bounds(lines, i)
                code = '\n'.join(lines[i:end])
                # try to detect methods
                methods = []
                class_lines = lines[i:end]
                for cl in class_lines:
                    method_match = re.match(r'\s*(?:async\s+)?([A-Za-z0-9_]+)\s*\(', cl)
                    if method_match:
                        methods.append(method_match.group(1))
                metadata = {
                    'file_path': str(p),
                    'type': 'class',
                    'name': name,
                    'methods': methods,
                    'start_line': i,
                    'end_line': end,
                    'language': 'js',
                    **project_info,
                }
                chunks.append((code, metadata))

        if not chunks:
            # Fallback: return the whole file as a single chunk
            metadata = {
                'file_path': str(p),
                'type': 'file',
                'name': p.stem,
                'language': Path(file_path).suffix.lstrip('.') or 'unknown',
                **project_info,
            }
            chunks.append((content, metadata))

        return chunks

    @staticmethod
    def parse_java_file(file_path: str, content: str, repo_root: Optional[str] = None) -> List[Tuple[str, Dict[str, Any]]]:
        """Heuristic Java parser that extracts classes and methods."""
        chunks: List[Tuple[str, Dict[str, Any]]] = []
        p = Path(file_path)
        repo = Path(repo_root) if repo_root else None
        project_info = CodeParser._extract_project_info(p, repo)

        lines = content.split('\n')
        class_re = re.compile(r'^\s*(?:public\s+)?(?:class|interface|enum)\s+([A-Za-z0-9_]+)')
        method_re = re.compile(r'^\s*(?:public|protected|private|static|final|synchronized|\s)+\s*[A-Za-z0-9_\<\>\[\]]+\s+([A-Za-z0-9_]+)\s*\(')

        for i, line in enumerate(lines):
            m = class_re.match(line)
            if m:
                name = m.group(1)
                end = CodeParser._find_block_bounds(lines, i)
                code = '\n'.join(lines[i:end])
                # find method names inside class block
                methods = []
                class_lines = lines[i:end]
                for cl in class_lines:
                    mm = method_re.match(cl)
                    if mm:
                        methods.append(mm.group(1))
                metadata = {
                    'file_path': str(p),
                    'type': 'class',
                    'name': name,
                    'methods': methods,
                    'start_line': i,
                    'end_line': end,
                    'language': 'java',
                    **project_info,
                }
                chunks.append((code, metadata))

        if not chunks:
            metadata = {
                'file_path': str(p),
                'type': 'file',
                'name': p.stem,
                'language': Path(file_path).suffix.lstrip('.') or 'unknown',
                **project_info,
            }
            chunks.append((content, metadata))

        return chunks

    @staticmethod
    def parse_file(file_path: str, content: str, repo_root: Optional[str] = None) -> List[Tuple[str, Dict[str, Any]]]:
        ext = Path(file_path).suffix.lower().lstrip('.')
        if ext == 'py':
            return CodeParser.parse_python_file(file_path, content, repo_root=repo_root)
        if ext in ('js', 'jsx', 'ts', 'tsx'):
            return CodeParser.parse_js_ts_file(file_path, content, repo_root=repo_root)
        if ext in ('java', 'kt'):
            return CodeParser.parse_java_file(file_path, content, repo_root=repo_root)

        # Fallback for other file types: return single file chunk with metadata
        p = Path(file_path)
        repo = Path(repo_root) if repo_root else None
        project_info = CodeParser._extract_project_info(p, repo)

        metadata = {
            'file_path': str(p),
            'type': 'file',
            'name': p.stem,
            'language': ext or 'unknown',
            **project_info,
        }
        return [(content, metadata)]
