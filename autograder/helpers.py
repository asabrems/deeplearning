import numpy as np

import json
import sys
import traceback

def assertions_all(user_vals, expected_vals, test_name, rtol=1e-5, atol=1e-8):
    if not assertions(user_vals, expected_vals, 'type', test_name, rtol=rtol, atol=atol):
        return False
    if not assertions(user_vals, expected_vals, 'shape', test_name, rtol=rtol, atol=atol):
        return False
    if not assertions(user_vals, expected_vals, 'closeness', test_name, rtol=rtol, atol=atol):
        return False
    return True

def assertions_no_type(user_vals, expected_vals, test_name, rtol=1e-5, atol=1e-8):
    if not assertions(user_vals, expected_vals, 'shape', test_name, rtol=rtol, atol=atol):
        return False
    if not assertions(user_vals, expected_vals, 'closeness', test_name, rtol=rtol, atol=atol):
        return False
    return True

def assertions(user_vals, expected_vals, test_type, test_name, rtol=1e-5, atol=1e-8):
    if test_type == 'type':
        try:
            assert type(user_vals) == type(expected_vals)
        except Exception as e:
            print('Type error, your type doesnt match the expected type.')
            print('Wrong type for %s' % test_name)
            print('Your type:   ', type(user_vals))
            print('Expected type:', type(expected_vals))
            return False
    elif test_type == 'shape':
        try:
            assert user_vals.shape == expected_vals.shape
        except Exception as e:
            print('Shape error, your shapes doesnt match the expected shape.')
            print('Wrong shape for %s' % test_name)
            print('Your shape:    ', user_vals.shape)
            print('Expected shape:', expected_vals.shape)
            return False
    elif test_type == 'closeness':
        try:
            assert np.allclose(user_vals, expected_vals, rtol=rtol, atol=atol)
        except Exception as e:
            print('Closeness error, your values dont match the expected values.')
            print('Wrong values for %s' % test_name)
            print('Your values:    ', user_vals)
            print('Expected values:', expected_vals)
            return False
    return True

def check_model_param_settings(model):
    """Checks that the parameters of a model are correctly configured.
    
    Note: again these tests aren't graded, although they will be next semester.

    Args:
        model (mytorch.nn.sequential.Sequential) 
    """
    # Iterate through layers and perform checks for each layer
    for idx, l in enumerate(model.layers):
        # Check that weights and biases of linear and conv1d layers are correctly configured
        if type(l).__name__ in ["Linear", "Conv1d"]:
            try:
                check_param_tensor(l.weight)
            except Exception:
                # If any checks fail print these messages index of the error printed below
                print(f"*WARNING: Layer #{idx} ({type(l).__name__}) has parameter (weight) tensor with incorrect settings:")
                print("\t" + str(sys.exc_info()[1]))
                print("\tNote: Your score on this test will NOT be affected by this message; this is to help you debug.")
                return False

            try:
                check_param_tensor(l.bias)
            except Exception:
                # If any checks fail print these messages index of the error printed below
                print(f"*WARNING: Layer #{idx} ({type(l).__name__}) has parameter (bias) tensor with incorrect settings:")
                print("\t" + str(sys.exc_info()[1]))
                print("\tNote: Your score on this test will NOT be affected by this message; this is simply a warning to help you debug future problems.")
                return False
        
        # Check batchnorm is correct
        elif type(l).__name__ == "BatchNorm1d":
            try:
                check_param_tensor(l.gamma)
            except Exception:
                # If any checks fail print these messages index of the error printed below
                print(f"*WARNING: Layer #{idx} ({type(l).__name__}) has gamma tensor with incorrect settings:")
                print("\t" + str(sys.exc_info()[1]))
                print("\tNote: Your score on this test will NOT be affected by this message; this is simply a warning to help you debug future problems.")
                return False
            
            try:
                check_param_tensor(l.beta)
            except Exception:
                # If any checks fail print these messages index of the error printed below
                print(f"*WARNING: Layer #{idx} ({type(l).__name__}) has beta tensor with incorrect settings:")
                print("\t" + str(sys.exc_info()[1]))
                print("\tNote: Your score on this test will NOT be affected by this message; this is simply a warning to help you debug future problems.")
                return False
            
            try:
                assert type(l.running_mean).__name__ == 'Tensor', f"Running mean param of BatchNorm1d must be a tensor. \n\tCurrently: {type(l.running_mean).__name__}, Expected: Tensor"
                assert type(l.running_var).__name__ == 'Tensor', f"Running var param of BatchNorm1d must be a tensor. \n\tCurrently: {type(l.running_var).__name__}, Expected: Tensor"
            except Exception:
                # If any checks fail print these messages index of the error printed below
                print(f"*WARNING: Layer #{idx} ({type(l).__name__}) has running mean/var tensors with incorrect settings:")
                print("\t" + str(sys.exc_info()[1]))
                print("\tNote: Your score on this test will NOT be affected by this message; this is simply a warning to help you debug future problems.")
                return False
        # TODO: Check that weights and biases of LSTM layers are correctly configured
        
    return True

def check_param_tensor(param):
    """Runs various (optional, ungraded) tests that confirm whether model param tensors are correctly configured

    Note: again these tests aren't graded, although they will be next semester.
    
    Args:
        param (Tensor): Parameter tensor from model
    """
    assert type(param).__name__ == 'Tensor', f"The param must be a tensor. You likely replaced a module param tensor on accident.\n\tCurrently: {type(param).__name__}, Expected: Tensor"
    assert isinstance(param.data, np.ndarray), f"The param's .data must be a numpy array (ndarray). You likely put a tensor inside another tensor somewhere.\n\tCurrently: {type(param.data).__name__}, Expected: ndarray"

    assert param.is_parameter == True, f"The param must have is_parameter==True.\n\tCurrently: {param.is_parameter}, Expected: True"
    assert param.requires_grad == True, f"The param must have requires_grad==True.\n\tCurrently: {param.requires_grad}, Expected: True"
    assert param.is_leaf == True, f"The param must have is_leaf==True.\n\tCurrently: {param.is_leaf}, Expected: True"

    if param.grad is not None:
        assert type(param.grad).__name__ == 'Tensor', f"If a module tensor has a gradient, the gradient MUST be a Tensor\n\tCurrently: {type(param).__name__}, Expected: Tensor"
        assert param.grad.grad is None, f"Gradient of module parameter (weight or bias tensor) must NOT have its own gradient\n\tCurrently: {param.grad.grad}, Expected: None"
        assert param.grad.grad_fn is None, f"Gradient of module parameter (weight or bias tensor) must NOT have its own grad_fn\n\tCurrently: {param.grad.grad_fn}, Expected: None"
        assert param.grad.is_parameter == False, f"Gradient of module parameter should NOT have is_parameter == True.\n\tCurrently: {param.is_parameter}, Expected: False"
        assert param.grad.shape == param.shape, f"The gradient tensor of a parameter must have the same shape as the parameter\n\tCurrently: {param.grad.shape}, Expected: {param.shape}"

def check_operation_output_settings(output, a, b=None, backpropped=False):
    """Checks that the output of a tensor operation and (optional) backprop over it is correctly configured.

    Note: This is not called anywhere in the main code.
          You may use it to debug your own code. 
    
    Args:
        output (Tensor): The result of a tensor operation (between 1 or 2 parents)
        a (Tensor): One parent that produced the output`  
        b (Tensor, optional): Another parent (optional, as some ops only have 1 parent)
        backpropped (bool, optional): If True, backprop occurred and gradients should be checked.
                                      If False, do not check for gradients
    """
    # determine if output should have requires_grad == True
    if b is not None:
        output_should_require_grad = a.requires_grad or b.requires_grad
    else:
        output_should_require_grad = a.requires_grad
    
    # Confirm that the output does have requires_grad == True
    assert output.requires_grad == True, f"If either parent requires a gradient, the child must also.\n\tCurrently: output.requires_grad == {output.requires_grad}, Expected: output.requires_grad == True"
    
    # Determine and confirm if output should be a leaf tensor    
    output_should_be_leaf = not output_should_require_grad
    assert output.requires_grad == True, f"If neither parent requires a gradient, the child must be a leaf tensor.\n\tCurrently: is_leaf == {output.is_leaf}, Expected: is_leaf == True"
    
    # If child should be an intermediate node, check that (won't check other types for now)
    if not output_should_be_leaf and output_should_require_grad:
        assert type(output.grad_fn).__name__ == "BackwardFunction", f"If an operation output is non-leaf and requires_grad, it must have a grad_fn.\n\tCurrently: grad_fn == {type(output.grad_fn).__name__}, Expected: BackwardFunction"

    # Check gradients if backprop occurred
    if backpropped:
        if b is not None:
            if b.requires_grad:
                assert type(b.grad).__name__ == "Tensor", f"One parent of operation required gradient, but did not accumulate it after backward.\n\tCurrently: parent.grad == {type(b.grad).__name__}, Expected: Tensor"
                assert b.grad.shape == b.shape, f"The shape of the parent's .grad and the parent's .data must be identical.\n\tCurrently: parent.grad.shape == {b.grad.shape}, Expected: {b.shape}"
            else:
                assert b.grad is None, f"Parent with requires_grad == False has a gradient accumulated, when it should not have.\n\tCurrently: parent.grad == {b.grad}, Expected: None"
        if a.requires_grad:
            assert type(a.grad).__name__ == "Tensor", f"One parent of operation required gradient, but did not accumulate it after backward.\n\tCurrently: parent.grad == {type(a.grad).__name__}, Expected: Tensor"
            assert a.grad.shape == a.shape, f"The shape of the parent's .grad and the parent's .data must be identical.\n\tCurrently: parent.grad.shape == {a.grad.shape}, Expected: {a.shape}"
        else:
            assert a.grad is None, f"Parent with requires_grad == False has a gradient accumulated, when it should not have.\n\tCurrently: parent.grad == {a.grad}, Expected: None"

def print_failure(cur_test, num_dashes=51):
    print('*' * num_dashes)
    print('The local autograder will not work if you do not pass %s.' % cur_test)
    print('*' * num_dashes)
    print(' ')

def print_name(cur_question):
    print(cur_question)

def print_outcome(short, outcome, point_value, num_dashes=51):
    score = point_value if outcome else 0
    if score != point_value:
        print("{}: {}/{}".format(short, score, point_value))
        print('-' * num_dashes)

def run_tests(tests, summarize=False):
    # calculate number of dashes to print based on max line length
    title = "AUTOGRADER SCORES"
    num_dashes = calculate_num_dashes(tests, title)

    # print title of printout
    print(generate_centered_title(title, num_dashes))

    # Print each test
    scores = {}
    for t in tests:
        if not summarize:
            print_name(t['name'])
        try:
            res = t['handler']()
        except Exception:
            res = False
            traceback.print_exc()
        if not summarize:
            print_outcome(t['autolab'], res, t['value'], num_dashes)
        scores[t['autolab']] = t['value'] if res else 0

    points_available = sum(t['value'] for t in tests)
    points_gotten = sum(scores.values())
    print("Total score: {}/{}\n".format(points_gotten, points_available))

    print("Summary:")
    print(json.dumps({'scores': scores}))

def calculate_num_dashes(tests, title):
    """Determines how many dashes to print between sections (to be ~pretty~)"""
    # Init based on str lengths in printout
    str_lens = [len(t['name']) for t in tests] + [len(t['autolab']) + 4 for t in tests]
    num_dashes = max(str_lens) + 1

    # Guarantee minimum 5 dashes around title
    if num_dashes < len(title) - 4:
        return len(title) + 10

    # Guarantee even # dashes around title
    if (num_dashes - len(title)) % 2 != 0:
        return num_dashes + 1

    return num_dashes

def generate_centered_title(title, num_dashes):
    """Generates title string, with equal # dashes on both sides"""
    dashes_on_side = int((num_dashes - len(title)) / 2) * "-"
    return dashes_on_side + title + dashes_on_side

def save_numpy_array(np_array, file_name):
    with open(file_name, 'wb') as f:
        np.save(f, np_array)

def load_numpy_array(file_path):
    with open(file_path, 'rb') as f:
        output = np.load(f, allow_pickle=True)
    return output