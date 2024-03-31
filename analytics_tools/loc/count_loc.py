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


def analyze_directory(directory, indent=""):
    line_counts = {"proc": 0, "other": 0}
    items = [item for item in os.listdir(directory) if item != "__pycache__"]
    for item in items:
        path = os.path.join(directory, item)
        if os.path.isdir(path):
            print(f"{indent}{item}/")
            sub_counts = analyze_directory(path, indent + "  ")
            line_counts["proc"] += sub_counts["proc"]
            line_counts["other"] += sub_counts["other"]
        elif item.endswith(".py"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    source = f.read()
                    counter = ProcCounter(source)
                    file_counts = {"proc": counter.proc_count, "other": counter.other_count}
                    line_counts["proc"] += file_counts["proc"]
                    line_counts["other"] += file_counts["other"]
                    print(f"{indent}{item}: @proc={file_counts['proc']}, Other={file_counts['other']}")
            except Exception as e:
                print(f"{indent}Error processing {path}: {e}")

    if directory != os.path.abspath(directory_path):  # Avoid printing for the root directory
        print(f"{indent}Total in '{os.path.basename(directory)}/': @proc={line_counts['proc']}, Other={line_counts['other']}")
    return line_counts


print("Project Line Counts:")
total_counts = analyze_directory(directory_path)
print(f"Total across all files and directories: @proc={total_counts['proc']}, Other={total_counts['other']}")
