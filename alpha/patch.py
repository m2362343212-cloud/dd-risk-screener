p = "alpha/fetch_prices_sina.py"
s = open(p).read()
if "with_timeout" in s:
    print("already patched"); raise SystemExit
rep = [
 ("import argparse, pathlib, random, time", "import argparse, pathlib, random, signal, time"),
 ("def with_retry(fn, tries=3, waits=(10, 60)):",
  "def _alarm(signum, frame):\n    raise TimeoutError(\"no answer in 90s\")\n\n\n"
  "def with_timeout(fn, sec=90):\n    signal.signal(signal.SIGALRM, _alarm)\n    signal.alarm(sec)\n"
  "    try:\n        return fn()\n    finally:\n        signal.alarm(0)\n\n\n"
  "def with_retry(fn, tries=3, waits=(10, 60)):"),
 ("            return fn()\n", "            return with_timeout(fn)\n"),
 ('    ap.add_argument("--sleep", type=float, default=2.0)\n',
  '    ap.add_argument("--sleep", type=float, default=2.0)\n'
  '    ap.add_argument("--part", default="1/1", help="e.g. 1/2 and 2/2 in two terminals")\n'),
 ("    OUT.mkdir(parents=True, exist_ok=True)\n",
  "    k, n = (int(x) for x in args.part.split(\"/\"))\n"
  "    codes = [c for j, c in enumerate(codes) if j % n == k - 1]\n"
  "    OUT.mkdir(parents=True, exist_ok=True)\n"),
]
for a, b in rep:
    assert s.count(a) == 1, a
    s = s.replace(a, b)
open(p, "w").write(s)
print("patched OK")
