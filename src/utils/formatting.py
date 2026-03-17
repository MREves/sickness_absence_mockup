
def debug_print(message):
    """
    This function will print messages only if DEBUG_MODE is set to True.
    """
    global DEBUG_MODE
    if DEBUG_MODE:
        print(message)