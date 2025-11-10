import threading

_keyboard_flag = False  # Internal flag


def get_keyboard_flag():
    return _keyboard_flag


def reset_keyboard_flag():
    global _keyboard_flag
    _keyboard_flag = False


def _callback(inp):
    global _keyboard_flag
    _keyboard_flag = True
    print("Pressed enter")


class KeyboardThread(threading.Thread):
    def __init__(self, input_cbk=_callback, name="keyboard-input-thread"):
        self.input_cbk = input_cbk
        super(KeyboardThread, self).__init__(name=name, daemon=True)
        self.start()

    def run(self):
        while True:
            self.input_cbk(input())