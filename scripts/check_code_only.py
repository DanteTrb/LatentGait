"""Check notebook hygiene and tracked outputs; --strip removes cell outputs only."""
import argparse
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strip',action='store_true')
    args=parser.parse_args()
    errors=[]
    for path in (ROOT/'notebooks').rglob('*.ipynb'):
        if '.ipynb_checkpoints' in path.parts:
            continue
        doc=json.loads(path.read_text())
        dirty=False
        for cell in doc.get('cells',[]):
            if cell.get('cell_type')=='code':
                dirty |= bool(cell.get('outputs')) or cell.get('execution_count') is not None
                if args.strip:
                    cell['outputs']=[];cell['execution_count']=None
                    for key in ('execution','ExecuteTime','collapsed','scrolled'):
                        cell.get('metadata',{}).pop(key,None)
        dirty |= bool(doc.get('metadata',{}).get('widgets'))
        if args.strip:
            doc.get('metadata',{}).pop('widgets',None)
            if dirty:
                path.write_text(json.dumps(doc,indent=1,ensure_ascii=False)+'\n')
        elif dirty:
            errors.append(str(path.relative_to(ROOT))+': saved outputs/execution metadata')
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    for name in filter(None,tracked):
        path=Path(name)
        if path.parts[0] in ('figures','results','outputs') or path.suffix.lower() in ('.pdf','.png','.nc','.csv','.xlsx','.pkl'):
            errors.append(name+': tracked generated/binary artifact')
    if errors:
        raise SystemExit('\n'.join(errors))
    print('PASS: notebook outputs empty; no tracked generated artifacts.')


if __name__=='__main__':
    main()
