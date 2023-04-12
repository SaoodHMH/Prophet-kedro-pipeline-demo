import pandas as pd
from prophet import Prophet

def set_carry_capacity(df: pd.DataFrame, carry_capacity):
    #When running Prophet, model will automatically detect 'cap' as carrying capacity
    #cap can be a single number or an array with length of df with carry capacity at each time instance
    if isinstance(carry_capacity, (int, float)):
        df['cap'] = carry_capacity
    elif len(carry_capacity) == len(df):
        df['cap'] = carry_capacity
    else:
        raise ValueError("carry_capacity must be a scalar or an array of the same length as df")
    return df

def set_saturating_min(df: pd.DataFrame, sat_min):
    #When running Prophet, model will automatically detect 'floor' as saturating minimum
    #floor can be a single number or an array with length of df with saturating minimum at each time instance
    #Prophet does not allow floor without cap
    if isinstance(sat_min, (int, float)):
        df['floor'] = sat_min
    elif len(sat_min) == len(df):
        df['floor'] = sat_min
    else:
        raise ValueError("sat_min must be a scalar or an array of the same length as df")
    return df

def run_prophet(df: pd.DataFrame, growth='linear', cap=None, floor=None):
    # Check if cap or floor are used without logistic growth
    if (growth != 'logistic') and ((cap is not None) or (floor is not None)):
        raise ValueError("cap and floor can only be used with logistic growth")
    
    # Check if floor is used without cap
    if (floor is not None) and (cap is None):
        raise ValueError("floor cannot be used without cap")
    
    # Check if floor is greater than or equal to cap
    if (floor is not None) and (floor >= cap):
        raise ValueError("floor must be less than cap")
    
    # Set cap value in dataframe
    if cap is not None:
        df = set_carry_capacity(df, cap)
    
    # Set floor value in dataframe
    if floor is not None:
        df = set_saturating_min(df, floor)

    # Create and train the Prophet model
    m = Prophet(growth=growth)
    m.fit(df)
    
    # Return the trained model
    return m

def forecast(model, period):
    #make a dataframe for period of forecasting
    future = model.make_future_dataframe(periods=period)
    #predict values on forecasting period
    forecast = model.predict(future)
    return forecast

def plot_forecast(model, forecast):
    fig = model.plot(forecast)
    return fig

def plot_components(model, forecast):
    fig = model.plot_components(forecast)
    return fig

