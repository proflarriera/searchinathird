# Search in a Third

Search in a Third is a Python package designed for optimization of hyperparameters in neural networks with an emphasis on moderate computational usage. It utilizes a greedy algorithm guided by heuristic directions that avoid traversing the entire multidimensional space of hyperparameters to achieve an optimal configuration of models efficiently.

## Features

- **Efficient Hyperparameter Optimization**: Focuses on reducing the number of models trained while still achieving optimal results.
- **Greedy Algorithm with Heuristic Guidance**: Narrows down the search space intelligently to find the best model configurations without exhaustive search.
- **Optimized Computational Use**: Designed to make the most out of available computational resources, avoiding unnecessary model training.

## Installation

You can install the package using pip:

```bash
pip install search_in_a_third
```

## Implementation Guide

### Introduction
This implementation guide is designed to quickly guide users through the implementation of Search in a Third.

### Installation and Library Import
To install the library, execute the following command:
```pip install search_in_a_third```

To import the library, you should include in your code:
```import search_in_a_third as siat```

### Data Loading and Processing
The first thing you need to do is process the data. For this, I will first give you some tips about the source files that will facilitate your work with search_in_a_third:
- Prioritize the use of files in csv format.
- Do not include the column names in your file.
- Be careful that the attribute separator is not included in the content of the columns of the file.
- Manage nulls before loading the file, the library executes a very basic management of nulls.

Once you have your file ready, you can include the following code in your program:
```data_model = siat.build_data_model(url, ['col1', 'col2', 'class'], 'class', 0.3, 'csv', ',')```

### Search Configuration
Preparing the configuration object is the most tedious part that involves the use of Search in a Third. You must be very careful in this step so that the algorithm understands what you want it to do.

The configuration object is a Python dictionary that must have the following mandatory elements:
- loss which must be a list with at least one valid loss function.
- optimize which must be a list with at least one valid optimization function.
- output_layer which must be a list with at least one valid activation function.
- learning_rate which must be a list of two numbers between 0 and 1.
- middle_layers_configuration with at least one Layer type element.

In turn, each element in middle_layers_configuration is a list that must have the following mandatory elements:
- At index 0 a guid type identifier.
- At index 1 a list of two integers that represent the minimum and maximum number of neurons to search for.
- At index 2 a list of two decimal numbers between 0 and 1 that represent the minimum and maximum dropout value to search for.
- At index 3 a list of valid activation functions.

An example of a configuration object is as follows:

```configuration_json = {'loss': ['categorical_crossentropy'], 'optimizer': ['Adam'], 'output_layer': ['sigmoid'], 'learning_rate': [0.0001, 0.1], 'middle_layers_configuration': [['51e2ebaa-1f44-41cd-b057-d56aaa42a13c', [1, 100], [0, 0.25], ['relu', 'sigmoid']], ['cdec3ae7-cd55-42bb-8894-2f11952488fc', [1, 100], [0, 0.25], ['relu', 'sigmoid']]]}```

### Execution of the Search

To execute the search for optimal configurations for your model's hyperparameters, you should include the following line of code in your program, where configuration_json is the configuration object you generated earlier, data_model is the data object you also generated earlier, and n is the number of iterations you want the algorithm to perform (I recommend starting with 3):


```iterations = siat.search(configuration_json, data_model, n)```

### Interpreting the Results
In the iterations variable, you will have a dictionary where each element represents each of the iterations. Each element will be named after the iteration index converted to String, for example: '0', '1', '2', ...

And each iteration will be a dictionary that in its 'results' element will have a list of result objects as follows:


```[{'id': '831865d9-cd6a-492b-a461-c823cce8644e', 'loss': 'categorical_crossentropy', 'model': '', 'result': 0.976190447807312, 'optimizer': 'Adam', 'output_layer': 'sigmoid', 'learning_rate': 0.06670000000000001, 'middle_layers': [{'id': '87b14a0a-eb15-42b5-a998-5f452d1ae622', 'units': 34, 'dropout': 0, 'activation': 'relu'}], 'x_characteristics': 4}, {…….]```


In the example, it is a model with an accuracy result of 0.976, using a Categorical Crossentropy loss function, an Adam optimization function with a learning rate of 0.0667, output activated by the Sigmoid function, and a single hidden layer with 34 neurons, without dropout, activated by the Relu function.

If you want to find in this list the element that corresponds to the best-performing model, you can use the following line of code where i corresponds to the integer index of the iteration:


```best_performer = iterations['i']['best_performer']```

And if you want to discover the most promising search space offered by the iteration for a next iteration, you can run the following code where i corresponds to the integer index of the iteration:

```next_configuration = iterations['i']['next_configuration']```



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

- Diego Larriera

For more information, please contact proflarriera@gmail.com.