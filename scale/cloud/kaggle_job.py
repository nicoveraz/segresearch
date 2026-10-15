"""One Kaggle session (GPU T4 x2) for the scaling runs of the new rules (local follow-up to #27).

Pushed by scale/cloud/kaggle_push.sh as a Kaggle script kernel, with the code-and-bundle archive attached as a
private dataset. In one session (Kaggle stops a session after about 12 hours):
  1. unpack the archive; rebuild the corpus (SHA-256 checked) and the masks of ARMS in /tmp (not persisted: the
     corpus, entropies and masks take ~25 GB, more than the 20 GB of /kaggle/working);
  2. run RUNS, two at a time (one per T4, fp16 with loss scaling), never starting a run that would not finish
     before STOP_HOURS;
  3. copy each finished run's result.json and log, the rules and the registry lines to /kaggle/working/results.
RUNS and ARMS are filled in by kaggle_push.sh.
"""
import glob
import os
import shutil
import subprocess
import sys
import time

RUNS = []                       # [(size, arm, seed)], set by kaggle_push.sh
ARMS = []                       # arms whose masks to build
STOP_HOURS = 11.0               # Kaggle ends sessions at ~12 h
HOURS = {"1m": 1.5, "12m": 2.5}  # conservative per-run estimates on one T4 (refined after the first session)

T0 = time.time()
OUT = "/kaggle/working/results"
W = "/tmp/work"
os.makedirs(OUT, exist_ok=True)
log = open(f"{OUT}/session.log", "a")


def say(msg):
    line = f"[{(time.time() - T0) / 3600:5.2f} h] {msg}"
    print(line, flush=True); log.write(line + "\n"); log.flush()


arch = glob.glob("/kaggle/input/*/pilot.tgz") + glob.glob("/kaggle/input/**/pilot.tgz", recursive=True)
assert arch, "attach the dataset that holds pilot.tgz"
subprocess.run(["tar", "xzf", arch[0], "-C", "/kaggle/working"], check=True)
code = "/kaggle/working/pilot"
os.chdir(code)
env = dict(os.environ, PYTHONPATH=code)
say(f"disk: {shutil.disk_usage('/tmp').free / 2**30:.0f} GB free in /tmp, {shutil.disk_usage('/kaggle/working').free / 2**30:.0f} GB in /kaggle/working")
say(f"GPUs: {subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()}")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyarrow"], check=False)

say("corpus and masks")
# the corpus (checked against the bundle's SHA-256) and the masks of ARMS only (prep.pod builds the pilot's arms)
subprocess.run([sys.executable, "-c", f"from scale import prep; prep.ARMS = {tuple(ARMS)!r}; prep.pod('bundle', {W!r})"],
               check=True, env=env)
shutil.copy(f"{W}/rules.json", OUT)
say("masks built")

queue = list(RUNS)
running = {}                    # gpu -> (Popen, run, start time)
while queue or running:
    for gpu in (0, 1):
        p = running.get(gpu)
        if p and p[0].poll() is not None:
            size, arm, seed = p[1]
            name = f"{size}_{arm}_s{seed}"
            ok = os.path.exists(f"{W}/runs/{name}/result.json")
            say(f"{'done' if ok else 'FAILED'} {name} on GPU {gpu} after {(time.time() - p[2]) / 3600:.2f} h")
            if ok:
                os.makedirs(f"{OUT}/runs/{name}", exist_ok=True)
                shutil.copy(f"{W}/runs/{name}/result.json", f"{OUT}/runs/{name}/")
            if os.path.exists(f"{W}/logs/{name}.log"):
                shutil.copy(f"{W}/logs/{name}.log", f"{OUT}/")
            del running[gpu]
        if gpu not in running and queue:
            size, arm, seed = queue[0]
            left = STOP_HOURS - (time.time() - T0) / 3600
            if HOURS.get(size, 3.0) > left:
                say(f"not starting {size} {arm} s{seed}: {left:.1f} h left"); queue.pop(0); continue
            queue.pop(0)
            os.makedirs(f"{W}/logs", exist_ok=True)
            f = open(f"{W}/logs/{size}_{arm}_s{seed}.log", "w")
            proc = subprocess.Popen([sys.executable, "-m", "scale.prep", "run", "bundle", W, size, arm, str(seed)],
                                    stdout=f, stderr=subprocess.STDOUT, env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)))
            running[gpu] = (proc, (size, arm, seed), time.time())
            say(f"start {size} {arm} s{seed} on GPU {gpu}")
    time.sleep(30)

if os.path.exists("results/registry/scale.jsonl"):
    shutil.copy("results/registry/scale.jsonl", f"{OUT}/registry_scale.jsonl")
shutil.rmtree("/kaggle/working/pilot", ignore_errors=True)      # keep the output small: results only
say(f"session done: {len(glob.glob(f'{OUT}/runs/*/result.json'))} results in the output")
