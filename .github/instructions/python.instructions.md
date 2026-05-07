---
description: 'Python coding conventions and guidelines'
applyTo: '**/*.py'
---

# Python Coding Conventions

## Python Instructions

- Write clear and concise comments for each function.
- Ensure functions have descriptive names and include type hints.
- Provide docstrings following **numpy docstring** conventions.
- Use the `typing` module for type annotations (e.g., `List[str]`, `Dict[str, int]`).
    - If the package has it's own type definitions, use those instead of the standard library ones.
- Break down complex functions into smaller, more manageable functions.

## General Instructions

- Always prioritize readability and clarity.
- For algorithm-related code, include explanations of the approach used.
- Write code with good maintainability practices, including comments on why certain design decisions were made.
- Handle edge cases and write clear exception handling.
- For libraries or external dependencies, mention their usage and purpose in comments.
- Use consistent naming conventions and follow language-specific best practices.
- Write concise, efficient, and idiomatic code that is also easily understandable.

## Code Style and Formatting

- Follow the **PEP 8** style guide for Python.
- Maintain proper indentation (use 4 spaces for each level of indentation).
- Ensure lines do not exceed 79 characters.
- Place function and class docstrings immediately after the `def` or `class` keyword.
- Use blank lines to separate functions, classes, and code blocks where appropriate.

## Edge Cases and Testing

- Always include test cases for critical paths of the application.
- Account for common edge cases like empty inputs, invalid data types, and large datasets.
- Include comments for edge cases and the expected behavior in those cases.
- Write unit tests for functions and document them with docstrings explaining the test cases.

## Performance Considerations
- Always pay attention to memory management
- Always prioritize code performance (speed)
- Never change the core functionality of the code when optimizing for performance, unless explicitly asked to do so.

## Commit message generation
- When generating commit messages, ensure they are well explicative and follow the format: `<mod> <type>: <subject>`, where:
    - `<mod>` is the name of the module, file or Class being changed, without the path or extension.
    - `<type>` is one of the following: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`.
    - `<subject>` is a description of the change, concise but informative enough to understand the change without looking at the code.
- When a new function is added, the commit message should include the name of the function and a brief description of its purpose.
- If `__version__` is updated, the commit message should include the new version number and a summary of the changes included in that version.

### Example of commit messages
- `iff_processing refactor: optimized mode matrix processing for generalized use cases and better performance`
- `ComputeReconstructor perf: improved performance of reconstructor computation by optimizing matrix operations and reducing redundant calculations`
- `iff_processing feat: added ``_squash_mode_matrix`` function to handle cases with shuffle disabled`

## Example of Proper Documentation

```python
def calculate_area(radius: float) -> float:
    """
    Calculate the area of a circle given the radius.
    
    Parameters
    ----------
    radius : float
        The radius of the circle.
    
    Returns
    -------
    float
        The area of the circle, calculated as π * radius^2.
    """
    import math
    return math.pi * radius ** 2
```
