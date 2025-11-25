"""
Command parser for Unix-style shell with pipes and I/O redirection.

Parses command lines into executable pipelines.
"""

import re
import shlex
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class RedirectType(Enum):
    """Types of I/O redirection."""
    OUTPUT = ">"           # Overwrite output
    APPEND = ">>"          # Append output
    INPUT = "<"            # Read input


@dataclass
class Redirection:
    """I/O redirection specification."""
    type: RedirectType
    target: str


@dataclass
class Command:
    """A single command in a pipeline."""
    name: str
    args: List[str]
    kwargs: Dict[str, Any]

    def __repr__(self):
        args_str = ' '.join(self.args)
        kwargs_str = ' '.join(f'--{k}={v}' for k, v in self.kwargs.items())
        parts = [self.name, args_str, kwargs_str]
        return ' '.join(p for p in parts if p)


@dataclass
class Pipeline:
    """A pipeline of commands connected by pipes."""
    commands: List[Command]
    input_redirect: Optional[Redirection] = None
    output_redirect: Optional[Redirection] = None
    background: bool = False

    def __repr__(self):
        cmd_str = ' | '.join(str(c) for c in self.commands)

        if self.input_redirect:
            cmd_str = f"{cmd_str} {self.input_redirect.type.value} {self.input_redirect.target}"

        if self.output_redirect:
            cmd_str = f"{cmd_str} {self.output_redirect.type.value} {self.output_redirect.target}"

        if self.background:
            cmd_str += " &"

        return cmd_str


class ShellParser:
    """
    Parse shell command lines into executable pipelines.

    Supports:
    - Unix-style pipes: cmd1 | cmd2 | cmd3
    - I/O redirection: > file, >> file, < file
    - Flag arguments: --flag, --key=value
    - Positional arguments
    - Quoted strings
    - Background jobs: command &
    """

    def __init__(self):
        self.pipe_pattern = re.compile(r'\|')
        self.redirect_pattern = re.compile(r'(>>?|<)')

    def parse(self, line: str) -> Optional[Pipeline]:
        """
        Parse a command line into a Pipeline.

        Args:
            line: Command line string

        Returns:
            Pipeline or None if empty/comment
        """
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith('#'):
            return None

        # Check for background job
        background = False
        if line.endswith('&'):
            background = True
            line = line[:-1].strip()

        # Extract I/O redirections
        input_redirect, output_redirect, line = self._extract_redirections(line)

        # Split by pipes
        pipe_parts = self.pipe_pattern.split(line)

        # Parse each command
        commands = []
        for part in pipe_parts:
            part = part.strip()
            if not part:
                continue

            cmd = self._parse_command(part)
            if cmd:
                commands.append(cmd)

        if not commands:
            return None

        return Pipeline(
            commands=commands,
            input_redirect=input_redirect,
            output_redirect=output_redirect,
            background=background
        )

    def _extract_redirections(self, line: str) -> Tuple[Optional[Redirection], Optional[Redirection], str]:
        """
        Extract I/O redirections from command line.

        Returns:
            (input_redirect, output_redirect, remaining_line)
        """
        input_redirect = None
        output_redirect = None

        # Find all redirection operators
        # Use regex to find them while preserving quotes
        parts = []
        current = []
        i = 0

        while i < len(line):
            # Check for redirection operators
            if i < len(line) - 1 and line[i:i+2] == '>>':
                # Append redirection
                parts.append(('text', ''.join(current).strip()))
                parts.append(('redirect', '>>'))
                current = []
                i += 2
            elif line[i] in '<>':
                parts.append(('text', ''.join(current).strip()))
                parts.append(('redirect', line[i]))
                current = []
                i += 1
            else:
                current.append(line[i])
                i += 1

        if current:
            parts.append(('text', ''.join(current).strip()))

        # Process parts to extract redirections
        result_line = []
        i = 0

        while i < len(parts):
            ptype, pvalue = parts[i]

            if ptype == 'redirect' and i + 1 < len(parts):
                # Next part should be the filename
                _, filename = parts[i + 1]
                filename = filename.split()[0] if filename else ''

                if pvalue == '<':
                    input_redirect = Redirection(RedirectType.INPUT, filename)
                elif pvalue == '>':
                    output_redirect = Redirection(RedirectType.OUTPUT, filename)
                elif pvalue == '>>':
                    output_redirect = Redirection(RedirectType.APPEND, filename)

                # Remove filename from next part
                rest = ' '.join(filename.split()[1:]) if filename else ''
                if rest:
                    result_line.append(rest)

                i += 2  # Skip both redirect and filename
            elif ptype == 'text' and pvalue:
                result_line.append(pvalue)
                i += 1
            else:
                i += 1

        return input_redirect, output_redirect, ' '.join(result_line)

    def _parse_command(self, cmd_str: str) -> Optional[Command]:
        """
        Parse a single command string into Command object.

        Examples:
            "search 100" -> Command(name='search', args=['100'], kwargs={})
            "sample 5 --strategy=diverse" -> Command(name='sample', args=['5'], kwargs={'strategy': 'diverse'})
            "filter --min-value 0.8" -> Command(name='filter', args=[], kwargs={'min-value': '0.8'})
        """
        try:
            # Use shlex to handle quoted strings properly
            tokens = shlex.split(cmd_str)
        except ValueError:
            # Fallback for malformed quotes
            tokens = cmd_str.split()

        if not tokens:
            return None

        name = tokens[0]
        args = []
        kwargs = {}

        i = 1
        while i < len(tokens):
            token = tokens[i]

            if token.startswith('--'):
                # Flag argument
                flag = token[2:]

                if '=' in flag:
                    # --key=value
                    key, value = flag.split('=', 1)
                    kwargs[key] = self._parse_value(value)
                elif i + 1 < len(tokens) and not tokens[i + 1].startswith('--'):
                    # --key value
                    key = flag
                    value = tokens[i + 1]
                    kwargs[key] = self._parse_value(value)
                    i += 1  # Skip next token
                else:
                    # --flag (boolean)
                    kwargs[flag] = True
            elif token.startswith('-') and len(token) > 1:
                # Short flag: -f
                # For now, treat as boolean
                kwargs[token[1:]] = True
            else:
                # Positional argument
                args.append(self._parse_value(token))

            i += 1

        return Command(name=name, args=args, kwargs=kwargs)

    def _parse_value(self, value: str) -> Any:
        """
        Parse a value string to appropriate type.

        Tries to convert to int, float, or bool. Falls back to string.
        """
        # Boolean values
        if value.lower() in ('true', 'yes', 'on'):
            return True
        if value.lower() in ('false', 'no', 'off'):
            return False

        # Try numeric
        try:
            # Try int first
            if '.' not in value and 'e' not in value.lower():
                return int(value)
            else:
                return float(value)
        except ValueError:
            pass

        # String
        return value


def split_multiple_commands(line: str) -> List[str]:
    """
    Split a line with multiple commands separated by semicolons.

    Example: "cmd1; cmd2; cmd3" -> ["cmd1", "cmd2", "cmd3"]

    Note: Respects quotes and doesn't split semicolons in strings.
    """
    commands = []
    current = []
    in_quotes = False
    quote_char = None

    for char in line:
        if char in ('"', "'"):
            if not in_quotes:
                in_quotes = True
                quote_char = char
            elif char == quote_char:
                in_quotes = False
                quote_char = None
            current.append(char)
        elif char == ';' and not in_quotes:
            if current:
                commands.append(''.join(current).strip())
                current = []
        else:
            current.append(char)

    if current:
        commands.append(''.join(current).strip())

    return [c for c in commands if c]
