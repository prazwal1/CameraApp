"""
Keyboard Handler
Manages keyboard input and key bindings.
"""

class KeyboardHandler:
    """Handles keyboard input and key bindings."""
    
    def __init__(self):
        """Initialize keyboard handler."""
        self.key_bindings = {}
        self.last_key = None
    
    def bind_key(self, key, callback):
        """
        Bind a key to a callback function.
        
        Args:
            key (str or int): Key to bind (character string or ASCII code)
            callback (callable): Function to call when key is pressed
        """
        if isinstance(key, str):
            key_code = ord(key.lower())
        else:
            key_code = key
        
        self.key_bindings[key_code] = callback
    
    def unbind_key(self, key):
        """
        Unbind a key.
        
        Args:
            key (str or int): Key to unbind
        """
        if isinstance(key, str):
            key_code = ord(key.lower())
        else:
            key_code = key
        
        if key_code in self.key_bindings:
            del self.key_bindings[key_code]
    
    def handle_key(self, key_code):
        """
        Handle a key press.
        
        Args:
            key_code (int): Key code from cv2.waitKey()
        """
        if key_code == -1 or key_code == 255:  # No key pressed
            return
        
        self.last_key = key_code
        
        # Execute callback if key is bound
        if key_code in self.key_bindings:
            try:
                self.key_bindings[key_code]()
            except Exception as e:
                print(f"Error executing callback for key {chr(key_code) if 32 <= key_code <= 126 else key_code}: {e}")
    
