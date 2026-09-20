"""Check full multimodal weights, text decision gradients and native vision."""
import json
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from qev.evaluation import inference_autocast
from qev.model import QevModel
from qev.tokenization import collate_encodings, encode_question

t = time.time()
model = QevModel.from_pretrained(revision="2fc06364715b967f1860aea9cf38778875588b17",
    device="cuda", dtype=torch.float32, max_length=1024, max_state=384)
tok = model.get_tokenizer()
model.enable_gradient_checkpointing()
items = [encode_question("The customer was charged twice and asks for a refund.",
    {"instr": "Who should handle this request?", "options": ["billing", "technical", "sales"]},
    tok, 1024, 384)]
batch = collate_encodings(items, tok.pad_token_id, "cuda")
with inference_autocast(model):
    logits = model(**batch)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0], device="cuda"))
loss.backward()
assert torch.isfinite(loss)
assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.head.parameters())
assert hasattr(model.foundation.model, "visual")
assert not any(p.requires_grad for p in model.foundation.model.visual.parameters())
assert not any(p.requires_grad for p in model.foundation.lm_head.parameters())
torch.cuda.synchronize()
print("QEV_SMOKE_OK", json.dumps({"seconds":time.time()-t,"loss":loss.item(),
    "trainable":sum(p.numel() for p in model.trainable_parameters()), "tokens":len(items[0]['ids']),
    "foundation":type(model.foundation).__name__}), flush=True)
model.eval()
with torch.inference_mode(), inference_autocast(model):
    print("BF16_EVAL", model(**batch).tolist(), flush=True)
processor = model.get_processor()
image = Image.new("RGB", (224, 224), "white")
ImageDraw.Draw(image).rectangle((48, 48, 176, 176), fill="red")
messages = [{"role":"user", "content":[{"type":"image", "image":image},
    {"type":"text", "text":"What color is the shape in the image? Answer in one word."}]}]
inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
    enable_thinking=False, return_dict=True, return_tensors="pt")
inputs = {k:v.to("cuda") if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
with torch.inference_mode(), inference_autocast(model):
    output = model.generate_native(**inputs, max_new_tokens=16, do_sample=False)
new = output[0, inputs["input_ids"].shape[-1]:].tolist()
native = {"prompt":"red square on white background", "input_tokens":inputs['input_ids'].shape[-1],
    "token_ids":new,"text":tok.decode(new,skip_special_tokens=True), "precision":"cuda bf16 autocast"}
Path("runs").mkdir(exist_ok=True)
Path("runs/native_baseline.json").write_text(json.dumps(native,indent=2))
print("QEV_NATIVE_VISION_OK",json.dumps(native),flush=True)
