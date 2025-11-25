# Makefile for LaTeX paper compilation

# Main document
MAIN = main

# LaTeX compiler
LATEX = pdflatex
BIBTEX = bibtex

# Output PDF
PDF = $(MAIN).pdf

# Auxiliary files
AUX_FILES = *.aux *.log *.bbl *.blg *.out *.toc *.lof *.lot *.nav *.snm *.vrb

.PHONY: all clean distclean help

# Default target
all: $(PDF)

# Build PDF with bibliography
$(PDF): $(MAIN).tex references.bib
	@echo "First LaTeX pass..."
	$(LATEX) $(MAIN).tex
	@echo "Running BibTeX..."
	$(BIBTEX) $(MAIN)
	@echo "Second LaTeX pass..."
	$(LATEX) $(MAIN).tex
	@echo "Third LaTeX pass..."
	$(LATEX) $(MAIN).tex
	@echo "PDF compiled successfully: $(PDF)"

# Clean auxiliary files
clean:
	@echo "Removing auxiliary files..."
	rm -f $(AUX_FILES)
	@echo "Clean complete."

# Remove all generated files including PDF
distclean: clean
	@echo "Removing PDF..."
	rm -f $(PDF)
	@echo "Distclean complete."

# Quick build (single pass, useful for previewing changes)
quick:
	@echo "Quick build (single pass)..."
	$(LATEX) $(MAIN).tex

# Show help
help:
	@echo "Available targets:"
	@echo "  all (default) - Build the PDF with bibliography"
	@echo "  clean         - Remove auxiliary files"
	@echo "  distclean     - Remove all generated files including PDF"
	@echo "  quick         - Quick build (single LaTeX pass, no BibTeX)"
	@echo "  help          - Show this help message"
