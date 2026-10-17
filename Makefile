# Thin wrapper over scripts/bg.py, which works without make (Windows has no make by default).
PY ?= python
BG = $(PY) scripts/bg.py

.PHONY: help splits estimate synth grade features evaluate report test synthetic real

help:
	@$(BG) --help

splits:     ; $(BG) splits
estimate:   ; $(BG) estimate
synth:      ; $(BG) synth
grade:      ; $(BG) grade $(ARGS)
features:   ; $(BG) features $(ARGS)
evaluate:   ; $(BG) evaluate $(ARGS)
report:     ; $(BG) report $(ARGS)
test:       ; $(BG) test
synthetic:  ; $(BG) synthetic
real:       ; $(BG) real $(ARGS)
