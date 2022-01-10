from pathlib import Path
import argparse


def file_size(path):
    # Return file/dir size (MB)
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / 1e6
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.glob("**/*") if f.is_file()) / 1e6
    else:
        return 0.0


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = (
        input if len(input) > 1 else ("blue", "bold", input[0])
    )  # color arguments, string
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]


def export_engine(file, half, workspace=4, verbose=False, prefix=colorstr("TensorRT:")):
    # YOLOv5 TensorRT export https://developer.nvidia.com/tensorrt
    try:
        import tensorrt as trt

        file = Path(file)
        onnx = file.with_suffix(".onnx")
        assert onnx.exists(), f"failed to export ONNX file: {onnx}"

        print(f"\n{prefix} starting export with TensorRT {trt.__version__}...")
        f = file.with_suffix(".engine")  # TensorRT engine file
        logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            logger.min_severity = trt.Logger.Severity.VERBOSE

        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        config.max_workspace_size = workspace * 1 << 30

        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx)):
            raise RuntimeError(f"failed to load ONNX file: {onnx}")

        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        print(f"{prefix} Network Description:")
        for inp in inputs:
            print(
                f'{prefix}\tinput "{inp.name}" with shape {inp.shape} and dtype {inp.dtype}'
            )
        for out in outputs:
            print(
                f'{prefix}\toutput "{out.name}" with shape {out.shape} and dtype {out.dtype}'
            )

        half &= builder.platform_has_fast_fp16
        print(f"{prefix} building FP{16 if half else 32} engine in {f}")
        if half:
            config.set_flag(trt.BuilderFlag.FP16)
        with builder.build_engine(network, config) as engine, open(f, "wb") as t:
            t.write(engine.serialize())
        print(f"{prefix} export success, saved as {f} ({file_size(f):.1f} MB)")

    except Exception as e:
        print(f"\n{prefix} export failure: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Convert .pth or .ckpt model to onnx.",
    )
    parser.add_argument("--onnx-path", default="nanodet.onnx", type=str, help="Path to .yml config file.")
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    args = parser.parse_args()
    export_engine(file=args.onnx_path, half=args.half)
