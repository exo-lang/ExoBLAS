from typing import List, Self
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


class LoC:
    name: str
    children: List[Self]
    alg_loc: int
    sched_loc: int
    c_loc: int

    def __init__(self, name, children=[], alg_loc=0, sched_loc=0, c_loc=0):
        self.name = name
        self.children = children
        self.alg_loc = alg_loc
        self.sched_loc = sched_loc
        self.c_loc = c_loc

        for child in children:
            self.alg_loc += child.alg_loc
            self.sched_loc += child.sched_loc
            self.c_loc += child.c_loc

    def __repr__(self):
        tokens = [
            f"Algorithms LoC = {self.alg_loc}",
            f"Schedule and Other LoC = {self.sched_loc}",
            f"C Sources and Headers LoC = {self.c_loc}",
        ]
        if self.name.endswith(".py"):
            return ",".join(tokens[:2])
        elif self.name.endswith(".c") or self.name.endswith(".h"):
            return ",".join(tokens[2:])
        else:
            return ",".join(tokens)


def analyze_directory(directory, indent=""):
    children = []
    items = [item for item in os.listdir(directory) if item != "__pycache__"]
    for item in items:
        path = os.path.join(directory, item)
        if os.path.isdir(path):
            print(f"{indent}{item}/")
            sub_counts = analyze_directory(path, indent + "  ")
            children.append(sub_counts)
        elif item.endswith(".py"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    source = f.read()
                    counter = ProcCounter(source)
                    file_loc = LoC(item, alg_loc=counter.proc_count, sched_loc=counter.other_count)
                    children.append(file_loc)
                    print(f"{indent}{item}: {file_loc}")
            except Exception as e:
                print(f"{indent}Error processing {path}: {e}")
        elif item.endswith(".c") or item.endswith(".h"):
            loc = count_lines_of_code(path)
            file_loc = LoC(item, c_loc=loc)
            children.append(file_loc)
            print(f"{indent}{item}: {file_loc}")

    dir_name = os.path.basename(directory)
    dir_loc = LoC(dir_name, children)

    if directory != os.path.abspath(directory_path):  # Avoid printing for the root directory
        print(f"{indent}Total in '{dir_name}/': {dir_loc}")
    return dir_loc


print("Project Line Counts:")
total_counts = analyze_directory(directory_path)
print(f"Total across all files and directories: {total_counts}")
