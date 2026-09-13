"""Check the hand-written fp32 Qwen3 against transformers' implementation."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from qwen import Qwen3, HF_DIR
torch.cuda.set_per_process_memory_fraction(0.10)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
msgs = [{"role": "user", "content": "Write a very short story about a lighthouse keeper."}]
s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
print(repr(s))
ids = tok(s, return_tensors="pt").input_ids.cuda()
print(ids.shape)
hf = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True, torch_dtype=torch.float32).cuda().eval()
m = Qwen3()
m.alloc(1, 256)
with torch.no_grad():
    ref = hf(ids).logits[0]
    mine = m.forward(ids, 0)[0]
    print("prefill max abs diff", (ref[-1] - mine).abs().max().item())
    # greedy decode 20 tokens with both
    seq = ids.clone(); cur = mine
    for t in range(20):
        nt = cur.argmax()[None, None]
        seq = torch.cat([seq, nt], 1)
        cur = m.forward(nt, seq.shape[1] - 1)[0]
    ref2 = hf(seq).logits[0, -1]
    print("after 20 decode steps max abs diff", (ref2 - cur).abs().max().item(), "logit scale", ref2.abs().max().item())
    print(tok.decode(seq[0, ids.shape[1]:]))
