"""Helpers for exporting COMSOL result plots as image files."""

from pathlib import Path
from typing import Literal, NotRequired, Sequence, TypedDict

import mph
import numpy as np


ViewPlane = Literal["xy", "yz", "xz"]


class PlotExportSpec(TypedDict):
    """Configuration for one COMSOL 3D plot-group export."""

    plot_group: str
    output_name: str
    feature: NotRequired[str]
    unit: NotRequired[str]
    color_range: NotRequired[tuple[float, float]]


def export_comsol_3d_plots(
    model_file: str | Path,
    plots: Sequence[PlotExportSpec],
    view_plane: ViewPlane = "yz",
    output_dir: str | Path | None = None,
    dataset_name: str | None = None,
    view_name: str = "View 1",
    image_size_px: tuple[int, int] = (1400, 900),
) -> dict[str, Path]:
    """Export configurable COMSOL 3D plot groups as separate PNG files.

    Each plot specification selects a COMSOL plot group and output filename
    stem. It may also select a child feature and set its unit and color range.
    """
    if view_plane not in {"xy", "yz", "xz"}:
        raise ValueError("view_plane must be 'xy', 'yz', or 'xz'")
    if not plots:
        raise ValueError("plots must contain at least one export specification")

    model_path = Path(model_file).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else model_path.parent
    )
    destination.mkdir(parents=True, exist_ok=True)

    width, height = image_size_px
    if width <= 0 or height <= 0:
        raise ValueError("image_size_px values must be positive")
    output_names = [spec["output_name"] for spec in plots]
    if len(output_names) != len(set(output_names)):
        raise ValueError("Every plot specification needs a unique output_name")
    outputs = {
        name: destination / f"{name}_3D_{view_plane}.png"
        for name in output_names
    }

    client = mph.start()
    model = None
    try:
        model = client.load(model_path)
        view = model / "views" / view_name
        if not view.exists():
            raise LookupError(f"COMSOL view does not exist: {view_name!r}")

        dataset = None
        if dataset_name is not None:
            dataset = model / "datasets" / dataset_name
            if not dataset.exists():
                raise LookupError(
                    f"COMSOL solution dataset does not exist: {dataset_name!r}"
                )

        coordinates = model.evaluate(
            ["x", "y", "z"],
            unit=["mm", "mm", "mm"],
            dataset=dataset,
            inner="last",
        )
        arrays = [np.asarray(value).ravel() for value in coordinates]
        center = np.array(
            [(values.min() + values.max()) / 2 for values in arrays]
        )
        distance = max(np.ptp(values) for values in arrays) * 5
        camera_settings = {
            "xy": (np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])),
            "yz": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
            "xz": (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
        }
        direction, up = camera_settings[view_plane]
        java_view = view.java_if_exists()
        camera_method = getattr(java_view, "camera", None)
        if camera_method is None:
            raise RuntimeError("The COMSOL 3D view does not expose a camera API")
        camera = camera_method()
        camera.set("projection", "orthographic")
        camera.set("viewscaletype", "automatic")
        camera.set("target", center.tolist())
        camera.set("rotationpoint", center.tolist())
        camera.set("position", (center + distance * direction).tolist())
        camera.set("up", up.tolist())

        for spec in plots:
            plot_group = model / "plots" / spec["plot_group"]
            if not plot_group.exists():
                raise LookupError(
                    f"COMSOL plot group does not exist: {spec['plot_group']!r}"
                )
            if dataset is not None:
                plot_group.property("data", dataset.tag())
            plot_group.property("view", view.tag())

            feature_name = spec.get("feature")
            unit = spec.get("unit")
            color_range = spec.get("color_range")
            if unit is not None or color_range is not None:
                if feature_name is None:
                    raise ValueError(
                        "A feature name is required when unit or color_range is set"
                    )
                feature = plot_group / feature_name
                if not feature.exists():
                    raise LookupError(
                        f"COMSOL result feature does not exist: {feature_name!r}"
                    )
                if unit is not None:
                    feature.property("unit", unit)
                if color_range is not None:
                    color_min, color_max = map(float, color_range)
                    if color_min >= color_max:
                        raise ValueError(
                            "color_range must be ordered from minimum to maximum"
                        )
                    feature.property("rangecoloractive", "on")
                    feature.property("rangecolormin", color_min)
                    feature.property("rangecolormax", color_max)

            output_file = outputs[spec["output_name"]]
            plot_group.run()
            image_export = (model / "exports").create(
                "Image3D", name=f"Export {plot_group.name()}"
            )
            image_export.property("sourceobject", plot_group.tag())
            image_export.property("imagetype", "png")
            image_export.property("pngfilename", output_file)
            image_export.property("size", "manualweb")
            image_export.property("unit", "px")
            image_export.property("width", float(width))
            image_export.property("height", float(height))
            image_export.property("options3d", "on")
            image_export.property("legend3d", True)
            image_export.property("title3d", True)
            image_export.property("logo3d", False)
            image_export.property("antialias", True)
            image_export.run()
    finally:
        if model is not None:
            client.remove(model)
        client.clear()
        client.disconnect()

    return outputs
