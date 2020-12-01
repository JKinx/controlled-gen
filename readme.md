## Instantiate Model

```ControlledGen(model_name, device)```

**model_name**

Name of the model to use. Currently, we only support `dateSet`.

**device**

Device to run the model on. It can be `gpu`, `cuda` or a specific cuda device such as `cuda:0`

## Get y, z from x  and z template (optional) `get_yz`

### Input

`x` : A tuple `(day, month, year)`

**Note:** Currently, the year is limted to the range 2000-2020


`template` (optional) : Template for `z` `[state_id/-1, (, ..., ),...,-1]`

The template is a list of state_ids (to force parts of `z`) and `-1` which stands for one or more of any state_id.

Parts of the template can be enclosed in parentheses `(...)` to allow for repetition of the part (one or more times).

**Note:** The template always needs to end with `-1`. No template is equivalent to the template `[-1]`


### Output

Dictionary object consisting of:

- `y` : Output sentence `[word0, word1, ...]`

- `score`: Score for sentence during beam search

- `z`: State for each word in y `[state0, state1, ...]`


## Get z from x and y `get_z`

### Input

`x` : A tuple `(day, month, year)`

**Note:** Currently, the year is limted to the range 2000-2020

`y`: Output sentence `[word0, word1, ...]`

### Output

Dictionary object consisting of:

- `z`: State for each word in y `[state0, state1, ...]`


## Testing the apis

To test the apis:

```python test_api.py --model model_name --device device --api api_name --template_id template_id```

**model_name:** dateSet

**device:** `cuda`, `gpu`, `cuda:0`, etc.

**api_name:** `get_yz`, `get_yz_templated`, `get_z`

**template_id:** only required if api is `get_yz_templated`. Integer between `0` and `7` for each format.

Example: For `x = (25,5,2003)`, the corresponding output formats are 

- `0` : `today is twenty five may 2003 .`
- `1` : `it is twenty five may of the year 2003 .` 
- `2` : `today is may twenty five , 2003 .`
- `3` : `it is may twenty five in the year 2003 .`
- `4` : `today is the twenty fifth of may , 2003 .`
- `5` : `it is the twenty fifth of may in the year 2003 .`
- `6` : `today is may the twenty fifth , 2003 .`
- `7` : `it is may the twenty fifth of the year 2003`