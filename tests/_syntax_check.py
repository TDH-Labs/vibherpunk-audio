import ast
import sys

for f in ("mobile_server.py", "hermes_voice.py", "desk_binding.py"):
    ast.parse(open(f).read(), filename=f)
    print("parsed", f)
print("syntax ok")
