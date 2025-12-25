# Introduction

You are tasked with generating solutions to python programming problems, which includes both code and some text describing the solution. The code generated should implement the problem provided and the text describing the solution should thoroughly explain what is being done. Your output should be in markdown format, with code blocks for the code and regular text for the explanations. Do not output wrapping "```markdown" or anything: just have your direct response be markdown content.

## General Structure

- **Introduction**: Brief summary of the problem and the approach taken.
- **Code Examples**: Multiple solutions provided, often starting with a basic solution and progressing to more advanced or optimized versions.
  - Each solution is labeled with a comment indicating its purpose or improvements.
  - Code blocks are clearly marked with language identifiers (e.g., `python`).
- **Explanation**: Detailed walkthrough of the code, including:
  - Why certain approaches were chosen.
  - Trade-offs between different solutions.
  - Common pitfalls and how to avoid them.
- **Bonuses**: Separate sections for each bonus, with solutions and explanations tailored to the additional requirements.

## Stylistic Elements

### Formatting
- **Code Blocks**: Use of triple backticks with language identifiers for syntax highlighting.
- **Comments**: Inline comments to explain key parts of the code.
- **Whitespace**: Consistent use of whitespace for readability.
- **Variable Naming**: Descriptive and meaningful variable names.

### Level of Detail
- **Step-by-Step Explanation**: Breakdown of the thought process behind each solution.
- **Comparisons**: Discussion of pros and cons of different approaches.
- **Edge Cases**: Consideration of edge cases and how they are handled.
- **Error Handling**: Explanation of how errors are managed and why certain exceptions are raised.

### Additional Features
- **Links to Documentation**: References to relevant Python documentation or external resources.
- **Examples**: Use of example inputs and outputs to illustrate the solution.
- **Bonuses**: Clear separation and detailed explanation of bonus solutions.

## Example Structure

```markdown
<problem>
**Title**: Brief description of the problem.

**Description**:
Detailed explanation of what needs to be accomplished, including examples and constraints.

**Bonuses**:
1. **Bonus 1**: Description of the first bonus challenge.
2. **Bonus 2**: Description of the second bonus challenge.
3. **Bonus 3**: Description of the third bonus challenge.

**Hints**:
- [Hint 1](link) "Explanation of the hint."
- [Hint 2](link) "Explanation of the hint."
</problem>

<solution>
**Introduction**:
Brief summary of the problem and the approach taken.

**Basic Solution**:
```python
# Code block with the basic solution
```

**Explanation**:
Detailed walkthrough of the basic solution, including why certain approaches were chosen and any trade-offs.

**Improved Solution**:
```python
# Code block with an improved solution
```

**Explanation**:
Detailed walkthrough of the improved solution, highlighting the improvements and any new trade-offs.

**Bonus 1**:
```python
# Code block with the solution for Bonus 1
```

**Explanation**:
Detailed walkthrough of the solution for Bonus 1, including any specific considerations or challenges.

**Bonus 2**:
```python
# Code block with the solution for Bonus 2
```

**Explanation**:
Detailed walkthrough of the solution for Bonus 2, including any specific considerations or challenges.

**Bonus 3**:
```python
# Code block with the solution for Bonus 3
```

**Explanation**:
Detailed walkthrough of the solution for Bonus 3, including any specific considerations or challenges.

**Additional Notes**:
Any additional insights, comparisons, or references to relevant documentation.
</solution>
```

## Key Points to Emulate

1. **Clarity**: Ensure that the problem and solution are clearly described and easy to follow.
2. **Completeness**: Provide multiple solutions and explain the reasoning behind each.
3. **Detail**: Include detailed explanations and considerations for edge cases and error handling.
4. **Organization**: Use a consistent structure with clear sections for the problem, solution, and bonuses.
5. **References**: Include links to relevant documentation or resources to aid understanding.

# Examples to Emulate

Here are some examples of existing problem and solution pairs, each in <example></example> tags with the problem inside <problem></problem> tags and the solution inside <solution></solution> tags. Use these examples as a guide for how to format your output and the level of detail required in the explanations:

{ examples }

# Your Task

Here is the new problem you need to solve:

<problem>
{ problem }
</problem>


