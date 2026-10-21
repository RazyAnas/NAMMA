# Kaggle script kernel: clone the repo and run the full video -> 3D pipeline; all output streams into the kernel log.
import subprocess, sys
subprocess.run('git clone -q --depth 1 -b claude/project-thread-l70z4u https://github.com/RazyAnas/NAMMA /kaggle/temp/NS && git -C /kaggle/temp/NS log --oneline -1', shell=True)
sys.exit(subprocess.call('bash /kaggle/temp/NS/colab/real_scenes/scripts/full_pipeline.sh', shell=True, stderr=subprocess.STDOUT))
