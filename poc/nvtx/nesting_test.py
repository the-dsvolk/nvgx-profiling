"""Does the SDK keep NVTX ranges of one domain nested inside a range of another domain?

Runs N iterations of: push "step i" (domain nvgx) > push "inner A" (default domain) > GPU work > pop >
push "inner B" (default domain) > GPU work > pop > pop. Mode "default-only" skips the nvgx range, mode
"same-domain" puts the inner ranges in nvgx too. Capture while it runs, then count ranges per name.
    python3 nesting_test.py [cross-domain|default-only|same-domain] [iterations]
"""

import sys
import time

import nvtx
import torch

mode = sys.argv[1] if len(sys.argv) > 1 else "cross-domain"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 600
x = torch.randn(512, 512, device="cuda")
torch.cuda.synchronize()
inner_domain = "nvgx" if mode == "same-domain" else None
for i in range(n):
    if mode != "default-only":
        nvtx.push_range(f"step {i}", domain="nvgx")
    nvtx.push_range("inner A", domain=inner_domain)
    y = x @ x
    nvtx.pop_range(domain=inner_domain)
    nvtx.push_range("inner B", domain=inner_domain)
    y = y.relu()
    nvtx.pop_range(domain=inner_domain)
    if mode != "default-only":
        nvtx.pop_range(domain="nvgx")
    torch.cuda.synchronize()
    time.sleep(0.01)
print(mode, "done", n)
