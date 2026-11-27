"""Demo Streamlit phân loại rác thải cho RQ1."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference.model_loader import (  # noqa: E402
    ModelBundle,
    apply_calibration,
    load_bundle,
    load_calibration,
)
from src.inference.predict import (  # noqa: E402
    denormalize,
    model_info,
    predict_proba,
    top_k,
)
from src.training.feature_maps import (  # noqa: E402
    CUSTOM_CNN_CONV_LAYERS,
    CUSTOM_CNN_POOL_LAYERS,
    collect_activations,
    plot_layer_maps,
)
from src.training.gradcam import GradCAM, overlay_cam  # noqa: E402
from src.utils.paths import experiments_root  # noqa: E402

RUNS = {
    "rq1_mobilenetv3s_full_p20_s42": "MobileNetV3_small pretrained",
    "rq1_customcnn_p20_s42": "Custom CNN",
    "rq1_mobilenetv3s_scratch_p20_s42": "MobileNetV3_small scratch",
}
MODEL_INFO = {
    "rq1_mobilenetv3s_full_p20_s42": {"gflops": "0,111", "latency": "9–11 ms"},
    "rq1_customcnn_p20_s42": {"gflops": "5,17", "latency": "26–29 ms"},
    "rq1_mobilenetv3s_scratch_p20_s42": {"gflops": "0,111", "latency": "9–11 ms"},
}
CLASS_VI = {
    "cardboard": "bìa carton",
    "glass": "thủy tinh",
    "metal": "kim loại",
    "paper": "giấy",
    "plastic": "nhựa",
}
MOBILENET_LAYERS = tuple(f"features.{index}" for index in range(13))


def _feature_layers(key: str) -> tuple[str, ...]:
    if key.startswith("rq1_customcnn"):
        return CUSTOM_CNN_CONV_LAYERS + CUSTOM_CNN_POOL_LAYERS
    return MOBILENET_LAYERS


@st.cache_resource
def get_bundle(key: str) -> ModelBundle:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = experiments_root() / "rq1" / key
    return load_bundle(key, RUNS[key], run_dir, device=device)


def _image_hash(image: Image.Image) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()[:16]


def _activations(bundle: ModelBundle, image: Image.Image) -> dict[str, torch.Tensor]:
    key = f"activations::{bundle.key}::{_image_hash(image)}"
    cached = st.session_state.get(key)
    if cached is not None:
        return cached
    tensor = bundle.transform(image).unsqueeze(0).to(bundle.device)
    activations = collect_activations(bundle.model, tensor, layers=_feature_layers(bundle.key))
    st.session_state[key] = activations
    return activations


def main() -> None:
    st.set_page_config(page_title="Phân loại rác thải", page_icon="♻️", layout="wide")
    st.title("Demo phân loại rác thải")
    st.caption(
        "RQ1 · Custom CNN / MobileNetV3_small pretrained / scratch · seed 42. "
        "Demo nghiên cứu, không phải sản phẩm."
    )

    calibration = load_calibration(ROOT / "configs" / "demo" / "calibration.yaml")
    with st.sidebar:
        st.header("Cấu hình")
        key = st.selectbox(
            "Mô hình",
            options=list(RUNS),
            format_func=lambda item: RUNS[item],
            index=0,
        )
        uploaded = st.file_uploader("Ảnh đầu vào", type=["jpg", "jpeg", "png", "webp"])
        bundle = get_bundle(key)
        apply_calibration({key: bundle}, calibration)
        threshold = st.slider(
            "Ngưỡng cảnh báo",
            min_value=0.50,
            max_value=0.99,
            value=float(bundle.threshold),
            step=0.01,
        )
        show_gradcam = st.checkbox("Grad-CAM", value=True)
        show_maps = st.checkbox("Feature map", value=True)
        info = model_info(bundle)
        st.divider()
        st.caption(
            f"{info['params']:,} tham số · {MODEL_INFO[key]['gflops']} GFLOPs · "
            f"{MODEL_INFO[key]['latency']} CPU\n\n"
            f"Temperature = {bundle.temperature:.3f} · threshold = {bundle.threshold:.2f}"
        )

    if uploaded is None:
        st.info("Tải một ảnh lên để bắt đầu.")
        return

    try:
        image = Image.open(uploaded).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Không đọc được ảnh: {exc}")
        return

    probabilities, logits = predict_proba(bundle, image)
    top = top_k(probabilities, bundle.class_names, k=3)
    predicted = top[0][0]
    confidence = top[0][1]
    tensor = bundle.transform(image).unsqueeze(0).to(bundle.device)

    left, right = st.columns([1, 1.15])
    with left:
        st.image(image, caption="Ảnh gốc", use_container_width=True)
        st.image(denormalize(tensor.cpu()), caption="Input model 224×224", use_container_width=True)
    with right:
        st.subheader("Dự đoán")
        st.metric(
            f"{CLASS_VI.get(predicted, predicted)}",
            f"{confidence:.1%}",
            delta=f"top-2: {CLASS_VI.get(top[1][0], top[1][0])} ({top[1][1]:.1%})",
            delta_color="off",
        )
        table = pd.DataFrame(
            [
                {
                    "Lớp": CLASS_VI.get(name, name),
                    "Xác suất": round(probability, 4),
                }
                for name, probability in top
            ]
        )
        st.dataframe(table, hide_index=True, use_container_width=True)
        if confidence < threshold:
            st.warning(
                "Độ tin cậy thấp hơn ngưỡng cảnh báo. Không nên dùng kết quả này "
                "để phân loại tự động."
            )
        st.caption(
            "Ảnh ngoài miền RealWaste/TrashNet có thể sai nhiều hơn test. "
            "Xem external GD trong báo cáo."
        )

    tabs = st.tabs(["Giải thích", "Từng lớp", "Số liệu"])
    with tabs[0]:
        if not show_gradcam:
            st.caption("Đã tắt Grad-CAM.")
        else:
            cam = GradCAM(bundle.model)
            try:
                target_index = bundle.class_names.index(predicted)
                heatmap, _ = cam.generate(tensor, target_index)
                overlay = overlay_cam(bundle.display(image), heatmap)
                st.image(
                    overlay,
                    caption=f"Grad-CAM cho lớp dự đoán: {CLASS_VI.get(predicted, predicted)}",
                    use_container_width=True,
                )
            finally:
                cam.remove()

    with tabs[1]:
        if not show_maps:
            st.caption("Đã tắt feature map.")
        else:
            activations = _activations(bundle, image)
            layer = st.selectbox("Chọn layer", list(activations))
            figure = plot_layer_maps(
                {layer: activations[layer]},
                max_channels=6,
                title=f"{layer} · shape {tuple(activations[layer].shape[1:])}",
            )
            st.pyplot(figure)
            plt.close(figure)
            st.caption(
                "6 kênh có phương sai cao nhất và map trung bình. "
                "Logic nằm trong src/training/feature_maps.py."
            )

    with tabs[2]:
        st.write("Logits (chưa calibration):")
        st.dataframe(
            pd.DataFrame(
                {
                    "Lớp": [CLASS_VI.get(name, name) for name in bundle.class_names],
                    "Logit": logits.numpy().round(4),
                    "Softmax": probabilities.round(4),
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            f"Thiết bị: {bundle.device} · Temperature: {bundle.temperature:.3f} · "
            f"Ngưỡng cảnh báo: {threshold:.2f}"
        )


if __name__ == "__main__":
    main()
