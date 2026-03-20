import runpy
% ...existing code...

if __name__ == "__main__":
    runpy.run_module("scripts.data_processing.extract_ntl_data", run_name="__main__")
