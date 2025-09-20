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
    
    def is_key_bound(self, key):
        """
        Check if a key is bound to a callback.
        
        Args:
            key (str or int): Key to check
            
        Returns:
            bool: True if key is bound, False otherwise
        """
        if isinstance(key, str):
            key_code = ord(key.lower())
        else:
            key_code = key
        
        return key_code in self.key_bindings
    
    def get_bound_keys(self):
        """
        Get all bound keys.
        
        Returns:
            list: List of bound key codes
        """
        return list(self.key_bindings.keys())
    
    def get_last_key(self):
        """
        Get the last pressed key.
        
        Returns:
            int or None: Last key code or None if no key pressed
        """
        return self.last_key
    
    def clear_bindings(self):
        """Clear all key bindings."""
        self.key_bindings.clear()
    
    def bind_multiple_keys(self, key_callback_pairs):
        """
        Bind multiple keys at once.
        
        Args:
            key_callback_pairs (list): List of (key, callback) tuples
        """
        for key, callback in key_callback_pairs:
            self.bind_key(key, callback)
    
    def get_key_help(self):
        """
        Get help text for all bound keys.
        
        Returns:
            list: List of help strings
        """
        help_lines = []
        for key_code in sorted(self.key_bindings.keys()):
            if 32 <= key_code <= 126:  # Printable ASCII
                key_char = chr(key_code).upper()
                help_lines.append(f"{key_char}: {self.key_bindings[key_code].__name__}")
        return help_lines