import nbformat
import re

def generate_toc(notebook_path, toc_placeholder='<!-- TOC -->'):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    headers = []
    for cell in nb.cells:
        if cell.cell_type == 'markdown':
            lines = cell.source.split('\n')
            for line in lines:
                header_match = re.match(r'^(#+)\s+(.*)', line)
                if header_match:
                    level = len(header_match.group(1))
                    title = header_match.group(2).strip()
                    # Create a URL-friendly anchor
                    anchor = re.sub(r'[^\w\s-]', '', title).lower()
                    anchor = re.sub(r'\s+', '-', anchor)
                    headers.append((level, title, anchor))
    
    # Build TOC markdown
    toc_md = '# Table of Contents\n\n'
    for level, title, anchor in headers:
        indent = '  ' * (level - 1)
        toc_md += f'{indent}- [{title}](#{anchor})\n'
    
    # Insert TOC into the notebook
    toc_inserted = False
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type == 'markdown' and toc_placeholder in cell.source:
            nb.cells.insert(idx + 1, nbformat.v4.new_markdown_cell(toc_md))
            toc_inserted = True
            break
    
    if not toc_inserted:
        # If placeholder not found, prepend TOC
        nb.cells.insert(0, nbformat.v4.new_markdown_cell(toc_md))
    
    # Write the updated notebook back to file
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"Table of Contents {'inserted after placeholder' if toc_inserted else 'added at the top'}.")

# Usage
if __name__ == "__main__":
    notebook_file = 'DL-and-LLMs-Basics.ipynb'  # Replace with your notebook file name
    generate_toc(notebook_file)