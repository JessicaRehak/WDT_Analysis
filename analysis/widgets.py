from ipywidgets import widgets
from IPython.display import display
from IPython.display import clear_output
import analysis.core as core
import analysis.fom as fom

def comparatorWidget(comparator):
    """ Makes a widget for a given comparator """

    def handle_plot(b):
        clear_output()
        comparator.plot(dropdown.value, fom = fom.value, cycle = xaxis.value)

    
    # Get key values
    labels = sorted([k for k in comparator.data[0].data[0].data.keys()])
    dropdown = widgets.Select(options = labels, description='Serpent Parameter', disabled = False)
    button = widgets.Button(description = "Plot")
    fom = widgets.RadioButtons(options=['FOM', 'Error'], description="What to plot")
    xaxis = widgets.RadioButtons(options=['Cycles', 'CPU Time'], description="X-axis")
    display(dropdown, button, fom, xaxis)
    button.on_click(handle_plot)                     

