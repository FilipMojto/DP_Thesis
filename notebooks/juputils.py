import inspect
from IPython.display import HTML, display, Code


DEF_HTML = HTML("""
<style>
.jp-CodeCell .highlight,
.output pre,
.output code {
    background-color: #111111 !important;
    color: #94d4d4 !important;
}
</style>
""")

def display_func(func, language: str = 'Python', html = DEF_HTML):
    display(html)
    source = inspect.getsource(func)
    display(Code(source, language=language, ))