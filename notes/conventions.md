# Conventions

This file documents the coding conventions used within this project.

## C++ (.cc, .h)

- Indentation is four spaces and opening braces are placed on the next line for functions and class methods.
- Braces on a separate line for classes, structs, enums, and other structural elements.
- Braces on the same line for control flow and namespaces.
- Anonymous namespaces can be used to contain functions and variables to the current translation unit.
- Closing braces for namespaces should include a comment with the name of the namespace.
- Use `#pragma once` for header guards, with standard library includes listed before local headers and separated by a blank line.
- Class names use PascalCase (e.g., `CPU`, `Bus`, `PPU`), and functions use snake_case (e.g., `tensor_attr_to_string`, `find_loaded_library_path`).
- Local variables, parameters, and struct fields use lower snake case (e.g., `byte_count`, `base_addr`).
- Member variables use lower snake case with a trailing underscore (e.g., `bar_`, `byte_count_`).
- Enum values use PascalCase names prefixed with k (e.g., `kFirst`, `kSecond`).
- Other constants and constexpr expressions can use snake_case.
- Use `std::shared_ptr` for shared ownership, with constructor injection for dependencies.
- Prefer `const` correctness for read-only methods and pass-by-value for small types while using references for larger objects or smart pointers.
- Comments may delineate sections and provide inline notes.
- Multi-line doc comments are permitted for classes and inline explanation comments for logic blocks.
- Use `auto` for local variables when the type is clear from context, especially with calculated addresses or values.
- Header include ordering. Place stdlib before local headers, blank-line separated.

```cpp
#pragma once

#include <memory>

#include "something.h"

class Bar;

namespace {

int helper()
{
    return 1;
}

}  // namespace

namespace example {

/**
* Example class using repository conventions
*/
class Foo
{
public:
    enum Status {
        kFirst  = 1 << 0,
        kSecond = 1 << 1
    };

    // single line comment
    explicit Foo(std::shared_ptr<Bar> bar);

    int32_t Count() const
    {
        auto total = kStart;
        if (enabled) {
            for (int32_t i = 0; i < 4; i++) {
                // loop body
                total += i;
            }
        } else {
          // not enabled
        }

        return total + helper();
    }

private:
    std::shared_ptr<Bar> bar_;

    static constexpr int32_t kStart = 3;
};

}  // namespace example
```
