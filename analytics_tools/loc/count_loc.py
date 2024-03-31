import ast
import os

script_directory = os.path.dirname(os.path.abspath(__file__))
parent_of_parent_directory = os.path.dirname(os.path.dirname(script_directory))
directory_path = os.path.join(parent_of_parent_directory, "src")


class ProcCounter(ast.NodeVisitor):
    def __init__(self, source):
        self.proc_count = 0
        self.docstring_count = 0
        self.import_count = 0  # Initialize count for import lines
        source = [line for line in source.split("\n") if line.strip() and not line.strip().startswith("#")]
        self.total_lines = len(source)
        self.visit(ast.parse("\n".join(source)))
        # Adjust total lines by subtracting proc, docstring, and import counts to get other lines
        self.other_count = self.total_lines - self.proc_count - self.docstring_count - self.import_count

    def visit_FunctionDef(self, node):
        proc_decorated = any(isinstance(decorator, ast.Name) and decorator.id == "proc" for decorator in node.decorator_list)
        if proc_decorated:
            func_start = node.lineno - len(node.decorator_list)  # Adjust for decorator lines
            func_end = node.end_lineno
            func_lines = func_end - func_start + 1
            self.proc_count += func_lines

            if ast.get_docstring(node):
                doc_lines = len(ast.get_docstring(node, clean=False).split("\n"))
                self.docstring_count += doc_lines

        self.generic_visit(node)

    def visit_Import(self, node):
        self.import_count += 1

    def visit_ImportFrom(self, node):
        self.import_count += 1

    def generic_visit(self, node):
        if isinstance(node, (ast.Module, ast.ClassDef)) and ast.get_docstring(node):
            doc_lines = len(ast.get_docstring(node, clean=False).split("\n"))
            self.docstring_count += doc_lines
        super().generic_visit(node)


def strip_comments_and_empty_lines(source):
    """Strip single-line and multi-line comments and empty lines from the source code."""
    # Remove single-line comments
    source = re.sub(r"//.*", "", source)
    # Remove multi-line comments
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    # Remove empty lines
    source = os.linesep.join([line for line in source.splitlines() if line.strip()])
    return source


def count_lines_of_code(file_path):
    """Count lines of code in a C source or header file, excluding comments and empty lines."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
            stripped_source = strip_comments_and_empty_lines(source)
            return len(stripped_source.splitlines())
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0


def analyze_directory(directory, indent=""):
    line_counts = {"Algorithms": 0, "Schedule + Other": 0, "C Source + Headers": 0}
    items = [item for item in os.listdir(directory) if item != "__pycache__"]
    for item in items:
        path = os.path.join(directory, item)
        if os.path.isdir(path):
            print(f"{indent}{item}/")
            sub_counts = analyze_directory(path, indent + "  ")
            line_counts["Algorithms"] += sub_counts["Algorithms"]
            line_counts["Schedule + Other"] += sub_counts["Schedule + Other"]
            line_counts["C Source + Headers"] += sub_counts["C Source + Headers"]
        elif item.endswith(".py"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    source = f.read()
                    counter = ProcCounter(source)
                    file_counts = {"Algorithms": counter.proc_count, "Schedule + Other": counter.other_count}
                    line_counts["Algorithms"] += file_counts["Algorithms"]
                    line_counts["Schedule + Other"] += file_counts["Schedule + Other"]
                    print(f"{indent}{item}: {line_counts}")
            except Exception as e:
                print(f"{indent}Error processing {path}: {e}")
        elif item.endswith(".c") or item.endswith(".h"):
            loc = count_lines_of_code(path)
            line_counts["C Source + Headers"] += loc

    if directory != os.path.abspath(directory_path):  # Avoid printing for the root directory
        print(f"{indent}Total in '{os.path.basename(directory)}/': {line_counts}")
    return line_counts


print("Project Line Counts:")
total_counts = analyze_directory(directory_path)
print(f"Total across all files and directories: {total_counts}")
