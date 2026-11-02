# Operating rules (to avoid permission prompts)

- Work only inside this folder. Do not use `$TEMP`, `/tmp` or other external paths:
  save the start time directly in `<RUN>/final/start_time.txt`.
- No brace expansion (`{a,b,c}`), `$(...)` substitutions, backticks or complex pipes
  in shell commands: the permission parser does not analyse them and asks for confirmation.
  To create folders use explicit paths, e.g.
  `mkdir -p RUN1/P RUN1/V RUN1/steps RUN1/errors RUN1/final RUN1/dashboard RUN1/scripts`.
- Prefer the Read / Write / Edit / Glob / Grep tools over `cat`, `head`, `echo >` etc.
- For multi-step operations (timestamps, metrics, summary files) write a script in
  `<RUN>/scripts/` and run it with `python <script>`.
