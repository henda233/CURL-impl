"""tee 控制台输出到文件：替换 sys.stdout 与 sys.stderr，内容原样写回原流，
同时逐条追加写入日志文件并 flush，中断或异常时文件内容不丢失。

用法：
    import log_tee
    log_tee.start(out_dir / "train_log.txt")   # 在首个 print 之前调用
    ...
    log_tee.stop()                              # 正常收尾时恢复原流
"""

import sys


class _Tee:
    def __init__(self, real, log):
        self.real = real
        self.log = log

    def write(self, text):
        self.real.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self):
        self.real.flush()

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def isatty(self):
        return self.real.isatty()

    def fileno(self):
        return self.real.fileno()

    @property
    def encoding(self):
        return self.real.encoding


_log = None
_orig = (None, None)


def start(path):
    global _log, _orig
    if _log is not None:
        return
    _log = open(path, "a", encoding="utf-8")
    _orig = (sys.stdout, sys.stderr)
    sys.stdout = _Tee(sys.stdout, _log)
    sys.stderr = _Tee(sys.stderr, _log)


def stop():
    global _log, _orig
    if _log is None:
        return
    sys.stdout, sys.stderr = _orig
    _log.close()
    _log = None
    _orig = (None, None)
