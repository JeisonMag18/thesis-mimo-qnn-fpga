import importlib, platform, shutil, sys
print("Python:", sys.version.replace("\n", " "))
print("Platform:", platform.platform())
for name in ["numpy","matplotlib","torch","brevitas","onnx","qonnx","finn"]:
    try:
        m = importlib.import_module(name)
        print(f"[OK] {name}: {getattr(m, '__version__', 'installed')}")
    except Exception as e:
        print(f"[--] {name}: unavailable ({type(e).__name__})")
for exe in ["docker","vivado","vitis_hls"]:
    p = shutil.which(exe)
    print(f"[{'OK' if p else '--'}] {exe}: {p or 'not found in PATH'}")
