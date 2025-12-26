You are tasked with analyzing a series of written solutions (written in markdown, with some python code blocks) to Python programming exercises. Your goal is to extract and specify common elements of style and structure used in the provided solutions. You should output a clear description of the written style and structure observed in the examples so as to instruct another LLM to produce similar content. Pay particular attention to common elements and style in the solution section, as that is what will end up being generated. Include specification of markdown formatting used (which headers), the level of detail in explanations, and any other relevant stylistic elements.

Provide your output in markdown, as it will be used as part of the prompt for another LLM, but don't use any headings (just basic bold/lists, etc.), and don't be overly verbose (roughly 3 paragraphs of text). Don't provide an pre-amble or any wrapping text: just directly output the markdown content that could be used as instructions for generating similar solutions. Remember that what you produce will be embedded in another larger markdown document (with its own heading) to instruct an LLM.

Here the solution examples, each in <example></example> tags with the text of the solution solution inside <solution></solution> tags.

{{ examples }}

