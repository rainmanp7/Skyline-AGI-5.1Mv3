
# memory_manager.py
# Created on Nov13 2024

```python
class MemoryManager:
    def __init__(self):
        self.working_memory = {}
        self.short_term_memory = {}
        self.long_term_memory = {}

    def working_memory(self, key, value):
        self.working_memory[key] = value
        return "Working Memory Initialized"

    def short_term_memory(self, key, value):
        self.short_term_memory[key] = value
        return "Short-Term Memory Initialized"

    def long_term_memory(self, key, value):
        self.long_term_memory[key] = value
        return "Long-Term Memory Initialized"

    def memory_consolidation(self):
    # Iterate through the working memory and short-term memory
    for key, value in self.working_memory.items():
        # Store the value in the long-term memory
        self.long_term_memory[key] = value

    for key, value in self.short_term_memory.items():
        # Store the value in the long-term memory
        self.long_term_memory[key] = value

    # Clear the working memory and short-term memory
    self.working_memory.clear()
    self.short_term_memory.clear()

    return "Memory Consolidation Activated"

    def memory_retrieval(self, key, memory_type):
        if memory_type == "working":
            return self.working_memory.get(key, None)
        elif memory_type == "short_term":
            return self.short_term_memory.get(key, None)
        elif memory_type == "long_term":
            return self.long_term_memory.get(key, None)
        else:
            return None
```