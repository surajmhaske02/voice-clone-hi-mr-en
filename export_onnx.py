"""Export OpenVoice V2 tone color converter to ONNX (+ INT8) for ARM boards, then benchmark."""
import os, sys, time, types
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "OpenVoice"))
sys.modules.setdefault("wavmark", types.SimpleNamespace(load_model=lambda: torch.nn.Identity()))
from openvoice.api import ToneColorConverter  # noqa: E402

CKPT = "OpenVoice/checkpoints_v2/converter"
OUT = "onnx"


class VC(torch.nn.Module):
    """spec [1,513,T] + src/tgt fingerprints [1,256,1] -> audio [1,1,T*256]"""

    def __init__(self, m, tau=0.3):
        super().__init__()
        self.m, self.tau = m, tau

    def forward(self, spec, src_se, tgt_se):
        m = self.m
        lens = torch.full((1,), spec.size(2), dtype=torch.long)
        z, m_q, logs_q, y_mask = m.enc_q(spec, lens, g=src_se, tau=self.tau)
        z_p = m.flow(z, y_mask, g=src_se)
        z_hat = m.flow(z_p, y_mask, g=tgt_se, reverse=True)
        return m.dec(z_hat * y_mask, g=tgt_se)


def main():
    os.makedirs(OUT, exist_ok=True)
    conv = ToneColorConverter(f"{CKPT}/config.json", device="cpu")
    conv.load_ckpt(f"{CKPT}/checkpoint.pth")
    m = conv.model.eval()
    for mod in m.modules():  # bake weight_norm into plain weights
        try:
            torch.nn.utils.remove_weight_norm(mod)
        except (ValueError, AttributeError):
            pass
    vc = VC(m).eval()
    spec = torch.randn(1, 513, 200); se = torch.randn(1, 256, 1)
    fp32 = f"{OUT}/openvoice_vc.onnx"
    torch.onnx.export(vc, (spec, se, se), fp32, input_names=["spec", "src_se", "tgt_se"],
                      output_names=["audio"], dynamic_axes={"spec": {2: "T"}, "audio": {2: "N"}},
                      opset_version=17, dynamo=False)
    # reference encoder (enrollment) export: spec [1,T,513] -> se [1,256]
    torch.onnx.export(m.ref_enc, (torch.randn(1, 400, 513),), f"{OUT}/openvoice_se.onnx",
                      input_names=["spec"], output_names=["se"], dynamic_axes={"spec": {1: "T"}},
                      opset_version=17, dynamo=False)

    from onnxruntime.quantization import quantize_dynamic, QuantType
    int8 = f"{OUT}/openvoice_vc_int8.onnx"
    quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])

    import onnxruntime as ort
    with torch.no_grad():
        ref = vc(spec, se, se).numpy()
    for path in (fp32, int8):
        for th in (4, 1):
            o = ort.SessionOptions(); o.intra_op_num_threads = th
            s = ort.InferenceSession(path, o, providers=["CPUExecutionProvider"])
            feed = {"spec": spec.numpy(), "src_se": se.numpy(), "tgt_se": se.numpy()}
            out = s.run(None, feed)[0]
            t = time.perf_counter()
            for _ in range(5):
                s.run(None, feed)
            ms = (time.perf_counter() - t) / 5 * 1000
            dur = 200 * 256 / 22050
            err = np.abs(out - ref).max()
            print(f"{os.path.basename(path):24s} {os.path.getsize(path)/1e6:5.1f} MB  threads={th}  "
                  f"{ms:5.0f} ms for {dur:.2f}s audio (RTF {ms/1000/dur:.3f})  max|diff| {err:.3f}")


if __name__ == "__main__":
    main()
