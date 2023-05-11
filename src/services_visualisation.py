from src.model import Model

def visualize_services_predictions(model : Model):
    """        
    `visualize_services_predictions` function

    Description:
        Visualizes real values and predictions of the model.

    Args:
        model (`Model`): Model used to predict values.
    
    Returns:
        `None`
    """
    print("\nSample of predictions:")
    print("Real value: \t\t Predicted value:")

    for i in range(10):
        current_instance = int(model.get_data().get_y_test().iloc[i])
        current_prediction = int(model.get_predictions()[i])
        str_current_instance = "".join(str(current_instance))
        str_current_prediction = "".join(str(current_prediction))
        if len(str_current_instance) == 1:
            str_current_instance = "0000"+str_current_instance
        if len(str_current_instance) == 2:
            str_current_instance = "000"+str_current_instance
        if len(str_current_instance) == 3:
            str_current_instance = "00"+str_current_instance
        if len(str_current_instance) == 4:
            str_current_instance = "0"+str_current_instance
        if len(str_current_prediction) == 1:
            str_current_prediction = "0000"+str_current_prediction
        if len(str_current_prediction) == 2:
            str_current_prediction = "000"+str_current_prediction
        if len(str_current_prediction) == 3:
            str_current_prediction = "00"+str_current_prediction
        if len(str_current_prediction) == 4:
            str_current_prediction = "0"+str_current_prediction
        print("{0} \t\t\t {1}".format(str_current_instance, str_current_prediction))