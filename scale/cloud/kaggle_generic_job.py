"""One Kaggle session (GPU T4 x2): novelty vs entropy patching outside math (scale/generic.py; local follow-up).

For each domain in DOMAINS: prepare corpus, entropy model, novelty table, eval-matched thresholds and masks in /tmp,
then train and score the 12M models (entropy10 and novelty, seeds 0 and 1) two at a time, one per T4. Results
(per-run JSON, rules.json, logs) go to /kaggle/working/results/generic/<domain>/.
"""
import glob
import os
import shutil
import subprocess
import sys
import time

DOMAINS = ["code", "prose"]                           # set by kaggle_push.sh generic (GEN_DOMAINS / GEN_RUNS)
RUNS = [("entropy10", 0), ("novelty", 0), ("entropy10", 1), ("novelty", 1)]
ENT_BUDGETS = ""                                      # extra entropy budgets, e.g. "14 17" (set by kaggle_push.sh)
NOVNET = ""                                           # "1": also build novelty_net (set by kaggle_push.sh)
T0 = time.time()
OUT = "/kaggle/working/results/generic"
os.makedirs(OUT, exist_ok=True)
log = open(f"{OUT}/session.log", "a")


def say(m):
    line = f"[{(time.time() - T0) / 3600:5.2f} h] {m}"
    print(line, flush=True); log.write(line + "\n"); log.flush()


arch = glob.glob("/kaggle/input/**/pilot.tgz", recursive=True)
subprocess.run(["tar", "xzf", arch[0], "-C", "/kaggle/working"], check=True)
code = "/kaggle/working/pilot"
os.chdir(code)
env = dict(os.environ, PYTHONPATH=code, SEGR_GENERIC_ENT_BUDGETS=ENT_BUDGETS, SEGR_GENERIC_NOVNET=NOVNET)
for dom in DOMAINS:
    W = f"/tmp/gen_{dom}"; o = f"{OUT}/{dom}"
    os.makedirs(o, exist_ok=True)
    say(f"{dom}: prep")
    r = subprocess.run([sys.executable, "-m", "scale.generic", "prep", dom, W], env=env, capture_output=True, text=True)
    open(f"{o}/prep.log", "w").write(r.stdout + r.stderr)
    if r.returncode != 0:
        say(f"{dom}: prep FAILED (see prep.log)")
        continue
    shutil.copy(f"{W}/rules.json", o); shutil.copy(f"{W}/corpus.json", o)
    say(f"{dom}: prep done; " + open(f"{W}/rules.json").read().replace("\n", " ")[:300])
    queue, running = list(RUNS), {}
    while queue or running:
        for gpu in (0, 1):
            p = running.get(gpu)
            if p and p[0].poll() is not None:
                rule, seed = p[1]
                f = f"{W}/results/{rule}_s{seed}.json"
                say(f"{dom}: {'done' if os.path.exists(f) else 'FAILED'} {rule} s{seed} after {(time.time() - p[2]) / 3600:.2f} h")
                if os.path.exists(f):
                    shutil.copy(f, o)
                del running[gpu]
            if gpu not in running and queue:
                rule, seed = queue.pop(0)
                lf = open(f"{o}/{rule}_s{seed}.log", "w")
                running[gpu] = (subprocess.Popen([sys.executable, "-m", "scale.generic", "run", dom, W, rule, str(seed)],
                                                 stdout=lf, stderr=subprocess.STDOUT,
                                                 env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu))),
                                (rule, seed), time.time())
                say(f"{dom}: start {rule} s{seed} on GPU {gpu}")
        time.sleep(30)
    shutil.rmtree(W, ignore_errors=True)
shutil.rmtree(code, ignore_errors=True)
say("session done")
