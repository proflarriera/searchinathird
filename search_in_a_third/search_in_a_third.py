import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import itertools, random, time, operator, uuid, os

valid_functions = pd.DataFrame([
    {'name':'units', 'place':'layer'}, 
    {'name':'dropout', 'place':'layer'}, 
    {'name':'activation', 'place':'layer'}, 
    {'name':'layer', 'place':'base'}, 
    {'name':'learning_rate', 'place':'base'},
    {'name':'optimizer', 'place':'base'},
    {'name':'loss', 'place':'base'},
])

param_configuration=[
    {'param_name':'middle_layers_configuration', 'param_behavior':'enter_and_iterate', 'function':None}, 
    {'param_name':'middle_layers', 'param_behavior':'enter_and_iterate', 'function':None}, 
    {'param_name':'output_layer', 'param_behavior':'pick_best_value', 'function':None}, 
    {'param_name':'loss', 'param_behavior':'pick_best_value', 'function':'loss'}, 
    {'param_name':'optimizer', 'param_behavior':'pick_best_value', 'function':'optimizer'}, 
    {'param_name':'learning_rate', 'param_behavior':'get_segment_limits', 'function':'learning_rate'}, 
    {'param_name':'layer_id', 'param_behavior':'asign_guid', 'function':None}, 
    {'param_name':'units', 'param_behavior':'get_segment_limits', 'function':'units'}, 
    {'param_name':'dropout', 'param_behavior':'get_segment_limits', 'function':'dropout'}, 
    {'param_name':'activation', 'param_behavior':'pick_best_value', 'function':'activation'}, 
    {'param_name':'id', 'param_behavior':'ignore_param', 'function':None}, 
    {'param_name':'x_characteristics', 'param_behavior':'ignore_param', 'function':None}, 
    {'param_name':'model', 'param_behavior':'ignore_param', 'function':None}, 
    {'param_name':'result', 'param_behavior':'ignore_param', 'function':None}, 
]

valid_loss = ['binary_crossentropy', 'categorical_crossentropy', 
                    'sparse_categorical_crossentropy', 'poisson', 'kl_divergence', 
                   'mean_squared_error', 'mean_absolute_error', 
                    'mean_absolute_percentage_error', 'mean_squared_logarithmic_error', 
                    'huber', 'log_cosh', 'hinge', 'squared_hinge', 'categorical_hinge']

valid_optimizer = ['SGD', 'RMSprop', 'Adam', 'AdamW', 'Adadelta', 
                        'Adagrad', 'Adamax', 'Adafactor', 'Nadam', 'Ftrl']

valid_activations = ['relu', 'sigmoid', 'softmax', 'softplus', 'softsign',
                         'tanh', 'selu', 'elu', 'exponential']

def find_middle_points(value_type, minimum, maximum):
    result = {'error': None, 'values': []}
    
    # Validación de tipos de entrada
    if not isinstance(value_type, str):
        result['error'] = 'The value type is not a valid value'
        return result
    
    if not isinstance(minimum, (int, float)) or not isinstance(maximum, (int, float)):
        result['error'] = 'Minimum and maximum values must be numeric'
        return result
    
    # Validación de lógica de rango
    if maximum <= minimum:
        result['error'] = 'Invalid range: maximum must be greater than minimum'
        return result
    
    # Validación de valores permitidos para value_type
    if value_type not in ['discret', 'dense']:
        result['error'] = 'Value type must be either "discret" or "dense"'
        return result
    
    # Cálculo de puntos medios según el tipo de valor
    try:
        if value_type == 'discret':
            if maximum - minimum == 1:
                first = minimum
                second = maximum
            elif maximum - minimum == 2:
                first = minimum + 1
                second = maximum
            else:
                first = round(minimum + ((maximum - minimum) / 3))
                second = round(minimum + ((maximum - minimum) / 1.5))
        else:  # Si no es discret, entonces es dense
            first = minimum + ((maximum - minimum) / 3)
            second = minimum + ((maximum - minimum) / 1.5)
        
        # Evitamos valores duplicados y ordenamos el resultado
        result['values'] = sorted(list(set([minimum, first, second, maximum])))
    
    except Exception as e:
        result['error'] = f'An error occurred: {str(e)}'
    
    return result

def dropout(minimum, maximum):
    # Validación de tipos de entrada
    if not isinstance(minimum, (int, float)) or not isinstance(maximum, (int, float)):
        return {'error': 'Minimum and maximum values must be numeric', 'values': []}
    
    # Validación de lógica de rango
    if not (0 <= minimum < maximum <= 1):
        return {'error': 'Minimum must be >= 0 and < maximum; maximum must be <= 1', 'values': []}
    
    # Uso de find_middle_points sin necesidad de try-except ya que maneja sus propios errores
    result = find_middle_points('dense', minimum, maximum)
    
    # Si find_middle_points no encontró errores devuelve los valores y si no devuelve el error
    return result

def activation(values):

    if not isinstance(values, (list, tuple)):  # Asegurar que values sea una lista o tupla
        return {'error': 'Input must be a list or tuple of activation functions', 'values': []}
    
    result = []
    invalid_values = []  # Almaceno valores inválidos para udevolver en el mensaje de error
    
    for value in values:
        if value not in valid_activations:
            invalid_values.append(value)
        else:
            result.append(value)
    
    if invalid_values:  # Si hay valores inválidos, devuelvo un error específico
        return {
            'error': f"One or more activation functions are not valid: {', '.join(invalid_values)}",
            'values': []
        }
    
    return {'error': None, 'values': list(dict.fromkeys(result))}  # Elimino duplicados manteniendo el orden


def units(minimum, maximum):
    result = {'error': None, 'values': []}

    if not isinstance(minimum, (int)) or not isinstance(maximum, (int)):
        result['error'] = 'Minimum and maximum values must be integer'
        return result
    
    # Validación de lógica de rango
    if maximum < minimum:
        result['error'] = 'Invalid range: maximum must be greater than minimum'
        return result

    return find_middle_points('discret', minimum, maximum)

        
def layer(id, layer_units, layer_dropout, activ):
    result = []

    punits = units(layer_units[0], layer_units[1])
    pdropout = dropout(layer_dropout[0], layer_dropout[1])
    pactivation = activation(activ)

    #Valido errores que vengan de las funciones de validación
    if punits['error'] or pdropout['error'] or pactivation['error']:
        return {'error': 'One or more parameters are invalid', 'values': []}

    for combinacion in itertools.product(punits['values'], 
                                         pdropout['values'], 
                                         pactivation['values']):
        result.append({'id':id, 'units':combinacion[0], 'dropout':combinacion[1], 'activation':combinacion[2]})   
    return result

def build_layers(layer_list):
    result = []
    for combinacion in itertools.product(*layer_list):
        result.append(combinacion)
    return result

def learning_rate(minimum, maximum):
    # Validación de tipos de entrada
    if not isinstance(minimum, (int, float)) or not isinstance(maximum, (int, float)):
        return {'error': 'Minimum and maximum values must be numeric', 'values': []}
    
    # Validación de lógica de rango
    if not (0 <= minimum < maximum <= 1):
        return {'error': 'Minimum must be >= 0 and < maximum; maximum must be <= 1', 'values': []}


    # Uso de find_middle_points sin necesidad de try-except ya que maneja sus propios errores
    result = find_middle_points('dense', minimum, maximum)
    
    # Si find_middle_points no encontró errores devuelve los valores y si no devuelve el error
    return result
    
def optimizer(values):
    
    if not isinstance(values, (list, tuple)):  # Asegurar que values sea una lista o tupla
        return {'error': 'Input must be a list or tuple of optimizer functions', 'values': []}

    result = []
    invalid_values = []  # Almaceno valores inválidos para devolver en el mensaje de error

    for value in values:
        if value not in valid_optimizer:
            invalid_values.append(value)
        else:
            result.append(value)
    
    if invalid_values:  # Si hay valores inválidos, devuelvo un error específico
        return {
            'error': f"One or more optimizer functions are not valid: {', '.join(invalid_values)}",
            'values': []
        }
    
    return {'error': None, 'values': list(dict.fromkeys(result))}  # Elimino duplicados manteniendo el orden y devuelvo la lista de resultados



def loss(values):
    if not isinstance(values, (list, tuple)):  # Asegurar que values sea una lista o tupla
        return {'error': 'Input must be a list or tuple of loss functions', 'values': []}

    result = []
    invalid_values = []  # Almaceno valores inválidos para devolver en el mensaje de error

    for value in values:
        if value not in valid_loss:
            invalid_values.append(value)
        else:
            result.append(value)
    
    if invalid_values:  # Si hay valores inválidos, devuelvo un error específico
        return {
            'error': f"One or more loss functions are not valid: {', '.join(invalid_values)}",
            'values': []
        }
    
    return {'error': None, 'values': list(dict.fromkeys(result))}  # Elimino duplicados manteniendo el orden y devuelvo la lista de resultados


def build_model(id, input_layer, middle_layers, output_layer, loss, optimizer_name, learning_rate, metrics):
    result = {'error':None,'model':None, 'id':id}
    first_layer = True
    try:
        model = keras.Sequential()
        for layer in middle_layers:
            if first_layer:
                if int(layer['units'])>0:
                    model.add(layers.Dense(int(layer['units']), activation=layer['activation'], input_shape=[input_layer]))
                    first_layer = False
            else:
                if int(layer['units'])>0:
                    model.add(layers.Dense(int(layer['units']), activation=layer['activation']))
            if int(layer['dropout']) > 0:
                model.add(layers.Dropout(layer['dropout']))
        
        optimizer = globals()[optimizer_name](learning_rate=learning_rate)
        
        model.add(layers.Dense(output_layer[0], output_layer[1]))
        model.compile(loss=loss, 
                      optimizer=optimizer, 
                      metrics=metrics)
        result['model'] = model
    except Exception as e:
        result['error'] = "Hubo un error al procesar el modelo: " + str(e)
    
    return result


def list_model_configurations(middle_layers, output_layers, loss, optimizer, learning_rates):
    #valido los parámetros de entrada
    if not isinstance(middle_layers, (list, tuple)):
        raise ValueError("middle_layers debe ser una lista")
    if not isinstance(output_layers, (list, tuple)):
        raise ValueError("output_layers debe ser una lista")
    if not isinstance(loss, (list, tuple)) or not all(isinstance(l, str) for l in loss):
        raise ValueError("loss debe ser una lista de strings")
    if not isinstance(optimizer, (list, tuple)) or not all(isinstance(opt, str) for opt in optimizer):
        raise ValueError("optimizer debe ser una lista de strings")
    if not (isinstance(learning_rates, (list, tuple)) and len(learning_rates) == 2 and all(isinstance(rate, (int, float)) for rate in learning_rates)):
        raise ValueError("learning_rates debe ser una lista de dos números (int o float)")

    result = []

    layer_choices = build_layers(middle_layers)
    if not isinstance(layer_choices, list):
        raise ValueError("build_layers debe devolver una lista")
    
    #compruebo que la función de validación de learning_rate devuelva valores válidos
    learning_rate_result = learning_rate(learning_rates[0], learning_rates[1])
    if learning_rate_result['error']:
        raise ValueError(f"Error en learning_rate: {learning_rate_result['error']}")
    elif not isinstance(learning_rate_result['values'], list):
        raise ValueError("learning_rate debe devolver un diccionario con una lista de valores bajo la clave 'values'")

    
    for combination in itertools.product(layer_choices, 
                                     output_layers, 
                                     loss, 
                                     optimizer, 
                                     learning_rate_result['values']
                                    ):
        result.append({'id':str(uuid.uuid4()), 
                       'middle_layers':combination[0], 
                       'output_layer':combination[1], 
                       'loss':combination[2], 
                       'optimizer':combination[3], 
                       'learning_rate':combination[4]}) 
    
    return result



def slice_data(filename, col_names, goal_name, test_size, file_type, separator):
    # Verificar que el archivo exista
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"El archivo {filename} no fue encontrado.")
    
    # Verifico que el tipo de archivo y el separador sean coherentes
    if file_type not in ['csv', 'data', 'txt']:  
        raise ValueError(f"Tipo de archivo {file_type} no soportado.")
    if not isinstance(separator, str):
        raise ValueError("El separador debe ser un string.")
    
    # Verificar que col_names sea una lista y tenga al menos un elemento
    if not isinstance(col_names, list) or len(col_names) < 1:
        raise ValueError("col_names debe ser una lista con al menos un elemento.")
    
    # Me aseguro que goal_name esté en col_names
    if goal_name not in col_names:
        raise ValueError(f"El nombre objetivo '{goal_name}' no está en los nombres de columnas proporcionados.")
    
    # test_size debe ser un flotante entre 0 y 1
    if not isinstance(test_size, float) or not 0 < test_size < 1:
        raise ValueError("test_size debe ser un flotante entre 0 y 1.")
    
    #controlo que no haya un error al cargar el archivo con pandas
    try:
        data = pd.read_csv(filename, names=col_names, sep=separator)
    except Exception as e:
        raise ValueError(f"Error al cargar el archivo: {e}")
    

    data=data.dropna()

    # Controlo que la columna objetivo esté en los datos
    try:
        X = data[data.columns[:-1]]
        Y = data[goal_name]
    except KeyError:
        raise KeyError(f"La columna objetivo '{goal_name}' no se encuentra en los datos.")

    # Verifico que no haya un error al dividir los datos
    try:
        seed = random.seed(int(time.time()))
        X_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    except Exception as e:
        raise ValueError(f"Error al dividir los datos: {e}")

    return {'X_train':X_train, 'x_test':x_test, 'y_train':y_train, 'y_test':y_test}


def build_data_model(filename, col_names, goal_name, test_size, file_type, separator):
    #traigo todos los parámetros validados de slice_data
    try:
        data = slice_data(filename, col_names, goal_name, test_size, file_type, separator)
    except Exception as e:
        raise Exception(f"Error en slice_data: {e}")
    
    #controlo que el escalado de valores numéricos no genere errores
    try:
        num = data['X_train'].select_dtypes(include=['int', 'float'])
        scaler = StandardScaler().fit(num)
    except Exception as e:
        raise Exception(f"Error al escalar datos numéricos: {e}")
    
    #controlo que no haya error al codificar los datos categóricos
    try:
        cat = data['X_train'].select_dtypes(include=['object'])
        if cat.empty:
            x_encoder = None
        else:
            x_encoder = OneHotEncoder(drop='if_binary').fit(cat)
    except Exception as e:
        raise Exception(f"Error al codificar datos categóricos: {e}")

    #controlo que no haya errores al codificar el subconjunto 'y'
    try:
        y_encoder = OneHotEncoder().fit(pd.DataFrame(data['y_train']))
    except Exception as e:
        raise Exception(f"Error al codificar el subconjunto 'y': {e}")
        
    #transform data
    X_train = pd.DataFrame([])

    if scaler is not None:
        try:
            X_train[num.columns] = scaler.transform(num).toarray()
        except:
            X_train[num.columns] = scaler.transform(num)

    if x_encoder is not None:
        try:
            X_train[x_encoder.get_feature_names_out()] = x_encoder.transform(cat).toarray()
        except Exception as e:
            raise Exception(f"Error al transformar datos categóricos en X_train: {e}")

    x_test = pd.DataFrame([])

    if scaler is not None:
        x_test[num.columns] = scaler.transform(data['x_test'].select_dtypes(include=['int', 'float']))
    if x_encoder is not None:
        x_test[x_encoder.get_feature_names_out()] = x_encoder.transform(data['x_test'].select_dtypes(include=['object'])).toarray()

    y_train = pd.DataFrame(y_encoder.transform(pd.DataFrame(data['y_train'])).toarray())
    y_test = pd.DataFrame(y_encoder.transform(pd.DataFrame(data['y_test'])).toarray())

    x_names = list(X_train.columns)
    x_characteristics = len(x_names)
    y_category_names = list(data['y_train'].unique())
    y_categories = len(y_category_names)

    return {'scaler':scaler, 'x_encoder':x_encoder, 'y_encoder':y_encoder,
           'X_train':X_train, 'x_test':x_test, 
           'y_train':y_train, 'y_test':y_test, 
           'x_names':x_names, 'x_characteristics':x_characteristics, 
           'y_category_names':y_category_names, 'y_categories':y_categories}


def iterate(configuration, data_model):
    returns = {}
    results = []
    errors = []

    required_fields = ['loss', 'optimizer', 'output_layer', 'learning_rate', 'middle_layers_configuration']
    
    # Verificar existencia de todos los campos requeridos en la configuración
    missing_fields = [field for field in required_fields if field not in configuration]
    if missing_fields:
        errors.append(f"Error: Faltan los siguientes campos de configuración requeridos: {', '.join(missing_fields)}.")

    # Valido que la configuración sea un objeto válido
    if (not 'loss' in configuration
        or not isinstance(configuration.get('loss', []), (list, tuple)) 
        or not all(loss in valid_loss for loss in configuration['loss'])):
        errors.append("'loss' debe ser una lista de strings válidos.")

    if (not 'optimizer' in configuration
        or not isinstance(configuration.get('optimizer', []), (list, tuple)) 
        or not all(optimizer in valid_optimizer for optimizer in configuration['optimizer'])):
        errors.append("'optimizer' debe ser una lista de strings válidos.")

    if (not 'output_layer' in configuration
        or not isinstance(configuration.get('output_layer', []), (list, tuple)) 
        or not all(output_layer in valid_activations for output_layer in configuration['output_layer'])):
        errors.append("'output_layer' debe ser una lista de strings.")

    if (not 'learning_rate' in configuration
        or not isinstance(configuration.get('learning_rate', []), (list, tuple)) 
        or not all(isinstance(learning_rate, (int, float)) for learning_rate in configuration['learning_rate'])):
        errors.append("'learning_rate' debe ser una lista de dos números (int o float).")

    if (not 'middle_layers_configuration' in configuration
        or not isinstance(configuration.get('middle_layers_configuration', []), (list, tuple)) 
        or not len(configuration['middle_layers_configuration'])>0):
        errors.append("'middle_layers_configuration' debe ser una lista de diccionarios con al menos un elemento.")

    if 'middle_layers_configuration' in configuration:
        middle_layers = []
        for i in range(len(configuration['middle_layers_configuration'])):
            if not isinstance(configuration['middle_layers_configuration'][i], (list, tuple)):
                errors.append(f"El elemento {i} de 'middle_layers_configuration' debe ser una lista.")
            
            # Valido que cada capa tenga la cantidad de neuronas y tipo correctos
            if not (isinstance(configuration['middle_layers_configuration'][i][1], (list, tuple)) 
                    and all(isinstance(size, int) for size in configuration['middle_layers_configuration'][i][1]) 
                    and len(configuration['middle_layers_configuration'][i][1])==2):
                errors.append(f"Los rangos de unidades de la capa {i} deben ser una lista de dos enteros que significan el máximo y el minimo donde buscar.")
            
            # Valido que cada capa tenga el rango de dropout correcto
            if not (isinstance(configuration['middle_layers_configuration'][i][2], (list, tuple)) 
                    and all(isinstance(dropout, (int, float)) for dropout in configuration['middle_layers_configuration'][i][2]) 
                    and len(configuration['middle_layers_configuration'][i][2])==2):
                errors.append(f"Los rangos de dropout de la capa {i} deben ser una lista de dos números que significan el máximo y el minimo donde buscar.")
            
            # Valido que cada capa tenga la función de activación correcta
            if not (isinstance(configuration['middle_layers_configuration'][i][3], (list, tuple)) 
                    and all(activation in valid_activations for activation in configuration['middle_layers_configuration'][i][3])):
                errors.append(f"Las funciones de activación de la capa {i} deben ser una lista de strings válidos.")


            # Construyo las capas intermedias validando que la funcion layer no devuelva errores
            ly = layer(*configuration['middle_layers_configuration'][i])
            if 'error' in ly:
                errors.append(f"Error en la capa {i}: {ly['error']}")
            else:
                middle_layers.append(ly)

    if errors:
        return {'error': str(errors) + """; Un ejemplo de configuración válida sería: 
                                        {'loss': ['categorical_crossentropy'], 'optimizer': ['Adam'], 'output_layer':
                                        ['sigmoid'], 'learning_rate': [0.06670000000000001, 0.1], 
                                        'middle_layers_configuration': [['8de1b496-3df2-495b-b8f4-cbc375d1dc18', [34, 67], [0, 0.08333333333333333], ['relu']]]
                                        }"""
                }
    
    try:
        model_configurations = list_model_configurations(
            middle_layers, 
            configuration['output_layer'], 
            configuration['loss'], 
            configuration['optimizer'], 
            configuration['learning_rate'])
        
        init = time.time()
        for model_configuration in model_configurations:
            model = build_model(model_configuration['id'],
                            data_model['x_characteristics'], 
                            model_configuration['middle_layers'], 
                            (data_model['y_categories'], model_configuration['output_layer']), 
                            model_configuration['loss'], 
                            model_configuration['optimizer'], 
                            model_configuration['learning_rate'], 
                            ['accuracy'])
            
            if model['error']:
                raise Exception(model['error'])

            history = model['model'].fit(data_model['X_train'], data_model['y_train'], epochs=10, validation_split=0.2)
            
            result = {'id':model_configuration['id'],
                            'x_characteristics':data_model['x_characteristics'], 
                            'middle_layers':model_configuration['middle_layers'], 
                            'output_layer':model_configuration['output_layer'],
                            'loss':model_configuration['loss'], 
                            'optimizer':model_configuration['optimizer'], 
                            'learning_rate':model_configuration['learning_rate'], 
                            'model':'', 
                            'result':max(history.history['accuracy'])}
            results.append(result)
            backend.clear_session()
        
        end = time.time()

        returns['configuration'] = configuration
        returns['model_configurations'] = model_configurations
        returns['results'] = results
        returns['best_performer'] = max(results, key=operator.itemgetter("result"))
        returns['process_time'] = end-init

        return returns
    except Exception as e:
        return f"Error al iterar para ejecutar la búsqueda: {str(e)}"



def get_neighbor_results(results, function, place, period, value):
    function_place = valid_functions[valid_functions['name']==function].iloc[0,:]['place']
    data = pd.DataFrame(results)

    for elem in data['middle_layers'][0]:
        for k,v in elem.items():  
            data[elem['id'] + '_' + k] = ''
    
    for i, item in data.iterrows():
        for elem in item['middle_layers']:
            for k,v in elem.items():  
                data[elem['id'] + '_' + k][i] = v

    better = max(results, key=operator.itemgetter("result"))
    condition_better_in_data=data.loc[:, 'id']==better['id']

    if function_place=='base':
        returns = data[[function, 'result']]
        data = data.drop(['id', 'middle_layers', 'result', 'model', 
                            function
                            ], axis=1)
    elif function_place=='layer':
        returns = data[[place + '_' + function, 'result']]
        data = data.drop(['id', 'middle_layers', 'result', 'model', 
                              place + '_id', 
                             place + '_' + function
                            ], axis=1)
    data['count'] = 0

    better_in_data = data.loc[condition_better_in_data]
    
    cols = list(data.columns)
    data = data.drop(better_in_data.index)
    
    for col in cols:
        if col!='count':
            data['count'] += data[col].isin(better_in_data[col])

    if function_place=='base':
        return returns.loc[data[data['count']==max(data['count'])].index, [function, 'result']]
    elif function_place=='layer':
        return returns.loc[data[data['count']==max(data['count'])].index, [place + '_' + function, 'result']]
    


def get_new_nodes(configuration, results, function, place, value):
    function_place = valid_functions[valid_functions['name']==function].iloc[0,:]['place']
    
    if function_place=='base':
        period = configuration[function]
    elif function_place=='layer':
        layers = pd.DataFrame(configuration['middle_layers_configuration'], columns=['id', 'units', 'dropout', 'activation'])
        period = layers[layers['id']==place].iloc[0,:][function]
    neighbors = get_neighbor_results(results, function, place, period, value)
    if function_place=='base':
        best_neighbor = neighbors[neighbors['result']==max(neighbors['result'])].iloc[0,:][function]
    elif function_place=='layer':    
        best_neighbor = neighbors[neighbors['result']==max(neighbors['result'])].iloc[0,:][place + '_' + function]

    if value<best_neighbor:
        return globals()[function](value, best_neighbor)
    return globals()[function](best_neighbor, value)



def pick_best_value(configuration, results, function, place, value):
    return [value]

def split_segment(configuration, results, function, place, value):
    return get_new_nodes(configuration, results, function, place, float(value))

def get_segment_limits(configuration, results, function, place, value):
    nodes = get_new_nodes(configuration, results, function, place, value)
    return (min(nodes['values']), max(nodes['values']))

def asign_guid(configuration, results, function, place, value):
    return str(uuid.uuid4())

def get_configuration_next_iteration(configuration, results, better_option, param_configuration):
    result = {'error':'', 'values':{}} 

    # Hago una validación inicial de tipos de los parámetros de entrada, 
    #no necesito validar los valores de los diccionarios que ya valido en en método iterate de donde me vienen
    if (not isinstance(configuration, dict) 
        or not isinstance(results, (list, tuple)) 
        or not isinstance(better_option, dict) 
        or not isinstance(param_configuration, (list, tuple))):
        return {"error": "Los parámetros de entrada no tienen los tipos correctos.", 'values':{}}

    try:
        config = pd.DataFrame(param_configuration)
        middle_layers_configuration=[]
        
        for key in better_option:

            action = config[config['param_name']==key].iloc[0,:]['param_behavior'] 
            if action == 'enter_and_iterate':
                for item in better_option[key]:
                    layer = []
                    
                    layer_id = str(uuid.uuid4())
                    layer.append(layer_id)
                    
                    for ikey in item:
                        action = config[config['param_name']==ikey].iloc[0,:]['param_behavior']
                        if action != 'ignore_param':
                            
                            param = f"{item['id']}_{ikey}"
                            function = config[config.loc[:, 'param_name']==ikey].iloc[0,:]['function']
                            if function is None:
                                place = None
                            else: 
                                place = valid_functions[valid_functions['name']==function].iloc[0,:]['place']      
                            value = item[ikey]
                            #layer.append(f"{action}({param},{place},{value})")
                            try:
                                place = param.split('_')[0]
                                print(param, action, function, place, value)
                                #function = param.split('_')[1]
                                layer.append(globals()[action](configuration, results, function, place, value)) #acá encontré como era para entender como llamar a la función get_new_nodes(configuration, resultados, 'units', 'b27b9f03-26df-4f8e-b8ed-5cf4d11a973c', 100)
                            except Exception as e:
                                print(param, e)
                    middle_layers_configuration.append(layer)
                result['values']['middle_layers_configuration'] = middle_layers_configuration
                
            else:
                if action != 'ignore_param':
                    #print(f"{action}({key},{better_option[key]})")
                    param = key
                    function = config[config.loc[:, 'param_name']==key].iloc[0,:]['function']
                    if function is None:
                        place = None
                    else: 
                        place = valid_functions[valid_functions['name']==function].iloc[0,:]['place']      
                    value = better_option[key]
                    result['values'][param] = globals()[action](configuration, results, param, place, value)

        return result
    except Exception as e:
        return {"error": f"Error en get_configuration_next_iteration: {str(e)}", 'values':{}}


def search(configuration, data_model, n_iterations):

    # Valido los parámetros de entrada
    # No necesito validar toda la configuraciuón porque ya la estoy validando en iterate
    if not isinstance(configuration, dict):
        raise ValueError("El parámetro configuración debe ser un diccionario.")
    
    if not isinstance(data_model, dict):
        raise ValueError("El parámetro data_model debe ser un diccionario.")
    
    if not isinstance(n_iterations, int) or n_iterations < 1:
        raise ValueError("El número de iteraciones debe ser un entero positivo.")

    iterations = {}
    errors = []
    
    for i in range(n_iterations):
        try:
            iter = iterate(configuration, data_model)
            if 'error' in iter:
                errors.append(iter['error'])
            else: 
                results = iter['results']

                mejor = max(results, key=operator.itemgetter("result"))

                next_config = get_configuration_next_iteration(configuration, results, mejor, param_configuration)
                configuration = next_config['values']
                if 'error' in configuration and configuration['error']:
                    errors.append(configuration['error'])

                iter['next_configuration'] = configuration
                iter['oedinal'] = i
                iterations[str(i)] = iter
        except Exception as e:
            errors.append(f"Error en la iteración {i}: {str(e)}")
    if errors:
        raise Exception(errors)
    else:
        return iterations