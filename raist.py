def compare_yolo_confs():
    files = [
        '/ML/ModH5/Benchmarks/Image/ObjectDetection/yolov10/runs/detect/train10_Mixup05_Mosaic05_fitnV2_SGD_150ep/args.yaml',
        '/ML/ModH5/Benchmarks/Image/ObjectDetection/yolov10/runs/detect/train10_Mixup05_Mosaic05__flipud05_fitnV2_SGD_150ep/args.yaml']

    def get_dyaml(f):
        import yaml
        with open(f) as stream:
            try:
                args = yaml.safe_load(stream)
                return args
            except yaml.YAMLError as exc:
                print(exc)

    a, b = get_dyaml(files[0]), get_dyaml(files[1])
    result = [(k, a[k], b[k]) for k in a if k in b and a[k] != b[k]]
    print(result)


exit()
