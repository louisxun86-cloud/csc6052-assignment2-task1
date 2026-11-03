"""Package verified submission artifacts; never runs experiments or submits work."""
import os
STUDENT_ID = os.environ.get('ASSIGNMENT_STUDENT_ID', 'YOUR_STUDENT_ID')
from pathlib import Path
import hashlib
import json
import zipfile
ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
OUT = ROOT / 'deliverables'
REPORT = ROOT / 'report'
PDF = WORKSPACE / 'output/pdf/Assignment2_Task1_ZhichaoXu_STUDENT_ID_OMITTED.pdf'.replace('STUDENT_ID_OMITTED', STUDENT_ID)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    cloud_record = OUT / '云端实际执行记录.txt'
    cloud_record.write_text('本次实际执行Notebook（保留真实开发、正式实验与汇总输出）：\nPRIVATE_COLAB_LINK_OMITTED\n\n浏览器原生Notebook下载未成功，因此未附声称为原始下载的本地执行本。\n完整逐请求日志已通过可见输出导出并校验SHA-256，保存在源码与日志ZIP以及自包含Notebook内。\n配套Task1_完整实验与结果.ipynb是根据冻结代码与真实日志构建的复现本，代码格未伪造执行记录。\n', encoding='utf-8')
    required = [PDF, OUT / 'Task1_完整实验与结果.ipynb', OUT / 'Task1_可复现源码与日志.zip', cloud_record]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    build = json.loads((OUT / 'build_manifest.json').read_text(encoding='utf-8'))
    formal = ROOT / 'actual_run/task1_assignment/formal/summary.json'
    summary = json.loads(formal.read_text(encoding='utf-8'))
    assert len(summary['methods']) == 4
    assert all((v['all_valid_inputs_finished'] for v in summary['methods'].values()))
    for path in (REPORT / 'generated').glob('*.tex'):
        if '\\Pending{' in path.read_text(encoding='utf-8'):
            raise ValueError(f'Unfilled report section: {path.name}')
    latex_zip = OUT / 'Task1_LaTeX报告源码.zip'
    latex_files = []
    for suffix in ('*.tex', '*.sty', '*.bst', '*.bib'):
        latex_files.extend(REPORT.glob(suffix))
    latex_files.extend((REPORT / 'generated').glob('*.tex'))
    if (REPORT / 'figures').exists():
        latex_files.extend((REPORT / 'figures').glob('*.pdf'))
        latex_files.extend((REPORT / 'figures').glob('*.png'))
    latex_files = sorted(set(latex_files))
    with zipfile.ZipFile(latex_zip, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in latex_files:
            archive.write(path, path.relative_to(REPORT).as_posix())
        archive.writestr('编译说明.txt', '使用完整TeX Live和XeLaTeX。解压后在main.tex所在目录执行两次：\nxelatex -interaction=nonstopmode -halt-on-error -no-shell-escape main.tex\n字体采用TeX Live自带TeX Gyre与Fandol。最终PDF已另附。\n')
    note = OUT / '提交说明.txt'
    note.write_text('Assignment 2 / Task 1\nZhichao Xu | STUDENT_ID_OMITTED | CSC 6052\n\n提交Blackboard的主文件：Assignment2_Task1_ZhichaoXu_STUDENT_ID_OMITTED.pdf。\n截止时间：2026年11月6日23:59（北京时间）。请在提交前通读报告并确认本人信息。\n本包已完成实验及报告材料，尚未替你上传或提交Blackboard。\n\nTask1_LaTeX报告源码.zip：报告完整可编译源码。\nTask1_可复现源码与日志.zip：冻结代码、数据、真实日志、图表和复现说明。\nTask1_完整实验与结果.ipynb：自包含复现本，含真实结果，默认关闭新的付费API调用。\n云端实际执行记录.txt：本次Colab实际执行本的链接与日志导出说明。\n原始100条数据中3条无选项；报告同时披露97题主结果、90题敏感性与100条保守得分。\n费用为按日志和声明费率计算的估计，不是平台账单。所有交付文件均不包含API Key。\n'.replace('STUDENT_ID_OMITTED', STUDENT_ID), encoding='utf-8')
    items = required + [latex_zip, note, OUT / 'Colab_完整实验结果.png']
    hashes = {p.name: {'sha256': digest(p), 'bytes': p.stat().st_size} for p in items}
    receipt = OUT / '提交文件校验.json'
    receipt.write_text(json.dumps(hashes, ensure_ascii=False, indent=2), encoding='utf-8')
    target = OUT / 'Assignment2_Task1_STUDENT_ID_OMITTED_提交包.zip'.replace('STUDENT_ID_OMITTED', STUDENT_ID)
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in items + [receipt]:
            archive.write(path, path.name)
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
    print(target)
    print('bytes', target.stat().st_size, 'sha256', digest(target))
if __name__ == '__main__':
    main()
