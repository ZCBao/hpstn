## Installation
1. Install [PyTorch](https://pytorch.org/get-started/previous-versions)
2. Install this library
    ```
    git clone https://github.com/ZCBao/hpstn.git
    cd hpstn
    pip install -e .
    ```

## Policy Test
```
cd hpstn
python policy/test.py --model_path output_dir/xxx --test_name YOUR_TEST_NAME --robot_circle_num 10 --robot_polygon_num 0 --use_gpu
```

## Reference
- [rl_rvo_nav](https://github.com/hanruihua/rl_rvo_nav)
- [intelligent-robot-simulator](https://github.com/hanruihua/intelligent-robot-simulator)