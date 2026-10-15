"""One Kaggle session (GPU T4 x2): train with novelty, patch with typed uncertainty at inference (local follow-up).

Builds the math corpus and the masks of ARMS in /tmp (as kaggle_job.py), trains 12M nov10 models (seeds 0 and 1,
one per T4) keeping their checkpoints, then scores each under every rule in CROSS at inference
(scale.prep cross). Its own rule (nov10) is included as a consistency check. Results go to
/kaggle/working/results/cross/.
"""
import glob
import os
import shutil
import subprocess
import sys
import time

ARMS = ("entropy10", "nov10", "typed14", "final+typed14")
CROSS = ["nov10", "entropy10", "typed14", "final+typed14"]
SEEDS = [0, 1]
T0 = time.time()
OUT = "/kaggle/working/results/cross"
W = "/tmp/work"
os.makedirs(OUT, exist_ok=True)
log = open(f"{OUT}/session.log", "a")


def say(m):
    line = f"[{(time.time() - T0) / 3600:5.2f} h] {m}"
    print(line, flush=True); log.write(line + "\n"); log.flush()


arch = glob.glob("/kaggle/input/**/pilot.tgz", recursive=True)
subprocess.run(["tar", "xzf", arch[0], "-C", "/kaggle/working"], check=True)
code = "/kaggle/working/pilot"
os.chdir(code)
env = dict(os.environ, PYTHONPATH=code, SEGR_SCALE_KEEP_CKPT="1")
say("corpus and masks")
subprocess.run([sys.executable, "-c", f"from scale import prep; prep.ARMS = {ARMS!r}; prep.pod('bundle', {W!r})"],
               check=True, env=env)
shutil.copy(f"{W}/rules.json", OUT)
say("masks built")
procs = []
for gpu, seed in enumerate(SEEDS):
    cmd = (f"{sys.executable} -m scale.prep run bundle {W} 12m nov10 {seed} && "
           f"{sys.executable} -m scale.prep cross bundle {W} 12m nov10 {seed} {' '.join(CROSS)}")
    lf = open(f"{OUT}/12m_nov10_s{seed}.log", "w")
    procs.append((subprocess.Popen(["bash", "-c", cmd], stdout=lf, stderr=subprocess.STDOUT,
                                   env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu))), seed))
    say(f"start 12m nov10 s{seed} (train, then cross-evaluate) on GPU {gpu}")
for p, seed in procs:
    p.wait()
    d = f"{W}/runs/12m_nov10_s{seed}"
    os.makedirs(f"{OUT}/12m_nov10_s{seed}", exist_ok=True)
    for f in ("result.json", "cross_eval.json"):
        if os.path.exists(f"{d}/{f}"):
            shutil.copy(f"{d}/{f}", f"{OUT}/12m_nov10_s{seed}/")
    say(f"{'done' if os.path.exists(f'{d}/cross_eval.json') else 'FAILED'} 12m nov10 s{seed}")
shutil.rmtree(code, ignore_errors=True)
say("session done")
