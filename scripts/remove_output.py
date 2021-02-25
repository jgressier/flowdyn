"""
Usage: python remove_output.py notebook.ipynb [ > without_output.ipynb ]
Modified from remove_output by Minrk

"""
import sys
import io
import os
from nbformat.current import read, write


def remove_outputs(nb):
    """remove the outputs from a notebook"""
    for ws in nb.worksheets:
        for cell in ws.cells:
            if cell.cell_type == 'code':
                cell.outputs = []
                for key in 'execution_count', 'prompt_number':
                    if key  in cell: del cell[key]

if __name__ == '__main__':
    for fname in sys.argv[1:]:
        bakname = fname+'.bak'
        os.rename(fname, bakname)
        with io.open(bakname, 'r') as f:
            nb = read(f, 'json')
        remove_outputs(nb)
        #base, ext = os.path.splitext(fname)
        with io.open(fname, 'w', encoding='utf8') as f:
            write(nb, f, 'json', version=4)
        print("wrote cleaned %s and backward copy %s" % (fname, bakname))