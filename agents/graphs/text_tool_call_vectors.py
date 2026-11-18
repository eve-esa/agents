"""Test vectors for :func:`agents.graphs.utils.parse_text_tool_calls`.

Plain data, no imports, so a consumer (the backend stream flush) can run the
same cases against its own parser. Each vector:

- ``name``: short case id;
- ``content``: the model reply text;
- ``tools``: names passed as ``allowed_names`` (``None``: no filter);
- ``expected``: ``[name, args]`` pairs in order; ``[]`` means no call, the
  text is an answer.
"""

TEXT_TOOL_CALL_VECTORS = [
    {
        "name": "name_and_object",
        "content": '[TOOL_CALLS]search{"q": "ndvi"}',
        "tools": None,
        "expected": [["search", {"q": "ndvi"}]],
    },
    {
        "name": "args_marker",
        "content": '[TOOL_CALLS]search[ARGS]{"q": "ndvi"}',
        "tools": None,
        "expected": [["search", {"q": "ndvi"}]],
    },
    {
        "name": "args_marker_with_spaces",
        "content": '[TOOL_CALLS] execute [ARGS] {"command": "ls"}',
        "tools": ["execute"],
        "expected": [["execute", {"command": "ls"}]],
    },
    {
        "name": "one_marker_per_call",
        "content": '[TOOL_CALLS]a[ARGS]{"x": 1}[TOOL_CALLS]b[ARGS]{"y": 2}',
        "tools": None,
        "expected": [["a", {"x": 1}], ["b", {"y": 2}]],
    },
    {
        "name": "json_array_live_trace",
        "content": '[TOOL_CALLS] [{"name": "execute", "arguments": {"command": "python compute_ndvi.py"}}]',
        "tools": ["execute"],
        "expected": [["execute", {"command": "python compute_ndvi.py"}]],
    },
    {
        "name": "json_array_several_calls_string_arguments",
        "content": (
            '[TOOL_CALLS] [{"name": "a", "arguments": {"x": 1}}, '
            '{"name": "b", "arguments": "{\\"y\\": 2}"}, {"name": "c", "args": {"z": 3}}]'
        ),
        "tools": None,
        "expected": [["a", {"x": 1}], ["b", {"y": 2}], ["c", {"z": 3}]],
    },
    {
        "name": "brace_inside_string_argument",
        "content": '[TOOL_CALLS]execute{"command": "echo \'}\' > out.txt"}',
        "tools": None,
        "expected": [["execute", {"command": "echo '}' > out.txt"}]],
    },
    {
        "name": "shell_escape_named_form",
        "content": '[TOOL_CALLS]execute[ARGS]{"command": "find . \\( -name x \\)"}',
        "tools": ["execute"],
        "expected": [["execute", {"command": "find . \\( -name x \\)"}]],
    },
    {
        "name": "shell_escape_json_array_form",
        "content": '[TOOL_CALLS] [{"name": "execute", "arguments": {"command": "ls \\( x"}}]',
        "tools": ["execute"],
        "expected": [["execute", {"command": "ls \\( x"}]],
    },
    {
        "name": "text_before_the_marker",
        "content": 'Let me check. [TOOL_CALLS]search{"q": "sar"}',
        "tools": ["search"],
        "expected": [["search", {"q": "sar"}]],
    },
    {
        "name": "prose_mentions_marker",
        "content": "The [TOOL_CALLS] token tells the server that a call follows.",
        "tools": None,
        "expected": [],
    },
    {
        "name": "prose_marker_then_bracket",
        "content": "Calls start with [TOOL_CALLS] [1] in the old format.",
        "tools": None,
        "expected": [],
    },
    {
        "name": "prose_word_then_object_unbound",
        "content": 'The [TOOL_CALLS] token {"a": 1} is a call.',
        "tools": ["search"],
        "expected": [],
    },
    {
        "name": "unbound_name_dropped",
        "content": '[TOOL_CALLS]search{"q": "a"}[TOOL_CALLS]nope{"q": "b"}',
        "tools": ["search"],
        "expected": [["search", {"q": "a"}]],
    },
    {
        "name": "bare_name_without_arguments",
        "content": "[TOOL_CALLS]search",
        "tools": ["search"],
        "expected": [],
    },
    {
        "name": "no_marker",
        "content": "No marker at all.",
        "tools": None,
        "expected": [],
    },
]
