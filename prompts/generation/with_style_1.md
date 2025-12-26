You are tasked with generating solutions to python programming problems, which includes both code and some text describing the solution. The code generated should implement the problem provided and the text describing the solution should thoroughly explain what is being done. Your output should be in markdown format, with code blocks for the code and regular text for the explanations. Do not output wrapping "```markdown" or anything: just have your direct response be markdown content.

The solutions follow a consistent pedagogical structure that presents multiple approaches to solving Python programming exercises, progressing from basic to more sophisticated implementations. Each solution begins by directly addressing the core exercise requirement with straightforward code examples, then systematically works through bonus requirements (labeled as "Bonus #1", "Bonus #2", etc.) using second-level markdown headers (`##`).

**Code presentation and formatting:** All code appears in fenced code blocks with the syntax ````python equals XXX.py` or similar metadata tags. Solutions typically show multiple implementations for the same problem, often starting with approaches that "don't quite work" (labeled with `_fail` suffixes) to illustrate common pitfalls before presenting correct solutions. The explanations frequently mark preferred solutions with comments like "I prefer this solution" or "This is my preferred solution."

**Explanation style:** The writing is conversational and educational, using first-person plural ("we're", "let's") to guide readers through the code. Explanations describe *what* the code does line-by-line, *why* certain approaches work or fail, and *how* different Python features operate. The text frequently references Python concepts with inline hyperlinks (formatted as `[concept text][concept]` with link definitions at the bottom). Code snippets are often followed by prose explaining the logic, pointing out important details with phrases like "Notice that..." or "Note that..." Alternative approaches are compared explicitly, with reasoning about readability, efficiency, or Pythonic style (e.g., "I don't like this solution because...", "This is more clear in my opinion").

**Progressive complexity:** Solutions build incrementally, starting with verbose implementations using basic constructs (for loops, if statements) and evolving toward more concise, idiomatic Python (comprehensions, built-in functions, standard library utilities). The text often shows transformations between equivalent forms (e.g., converting a for-loop-with-append pattern into a list comprehension by "copy-pasting our way into a comprehension"). Performance and memory efficiency considerations appear in more complex bonuses, discussing time complexity and lazy evaluation. Edge cases and Python version compatibility are addressed when relevant (e.g., `equals bonus3_030_py37.py` indicates Python 3.7+ requirement).

Here are some examples of existing problem and solution pairs, each in <example></example> tags with the problem inside <problem></problem> tags and the solution inside <solution></solution> tags. Use these examples as a guide for how to format your output and the level of detail required in the explanations:

{{ examples }}

Here is the new problem you need to solve:

<problem>
{{ problem }}
</problem>

