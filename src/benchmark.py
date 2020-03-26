from ptflops import get_model_complexity_info


def show_flops(model, input_shape=(3, 32, 32)):
    flops, params = get_model_complexity_info(model, input_shape, as_strings=True, print_per_layer_stat=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))


def get_flops(model, input_shape=(3, 32, 32)):
    flops, params = get_model_complexity_info(model, input_shape, as_strings=True, print_per_layer_stat=False)

    return flops, params
