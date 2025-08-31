Here is the structure of this module, please read in order:

1. `1_get_activations.py`: Get the activations of all generated tokens at generation time
2. `2_compute_directions.ipynb`: Compute and store the candidate directions
    - Select positive and negative activations:
        - [x] At `<think>` and `</think>`
        - [ ] At tokens surrounding the start and end
    - Compute candidate directions:
        - [x] Difference of mean
        - [ ] SVM (TODO)
        - [ ] Sequential steering by layer: select -> steer -> select -> steer -> ...
3. `3_select_steering_direction.ipynb`: Select the best candidate as the steering direction
    - Method:
        - [x] Max of mean cosine similarities
4. `4_apply_steering.ipynb`: Apply steering
5. `5_control_reasoning_along_token_dim.ipynb`: Develop scheduler